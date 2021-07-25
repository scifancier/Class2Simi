import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys, os
import tools
import dataloader
from criterion import forward_MCL, reweight_MCL
from learner import Learner_Likelihood
from pairwise import Class2Simi
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from transformer import transform_train, transform_test, transform_target
import Lenet
import ResNet26

parser = argparse.ArgumentParser()
parser.add_argument('--r', type=float, default=0.2, help="The start noise rate")
parser.add_argument('--bar', type=float, default=0.08, help="running experiment at noise rate [args.r, args.r + args.bar]")
parser.add_argument('--loss', type=str, default='forward', help="forward or reweight")
parser.add_argument('--n', type=int, default=2, help="the number of runs with random seeds")
parser.add_argument('--d', type=str, default='output', help="description for the output dir")
parser.add_argument('--p', type=int, default=1, help="0 for printing in terminal, 1 for printing in file")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--lr_es', type=float, default=0.001, help="Estimate learning rate")
parser.add_argument('--estimate_epochs', type=int, default=20, help="End epoch for estimating stage")
parser.add_argument('--epochs', type=int, default=120, help="End epoch")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--prob_dir', type=str, help='dir to save output probability files', default='prob')
parser.add_argument('--matrix_dir', type=str, help='dir to save estimated matrix', default='matrix')
parser.add_argument('--dataset', type=str, default='mnist', help="mnist, cifar10")
parser.add_argument('--num_workers', type=int, default=16, help="#Thread for dataloader")
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
parser.add_argument('--optimizer_es', type=str, default='Adam', choices=['Adam', 'SGD', 'RMSprop'])
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--gpu', type=str, default='n')
parser.add_argument('--noise_type', type=str, default='s', help="s or as")

args = parser.parse_args()
args.data_dir = './data'

if args.gpu != 'n':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(os.environ["CUDA_VISIBLE_DEVICES"])


def loaddata(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.dataset == 'mnist':
        train_data = dataloader.mnist_dataset(True, dir=args.data_dir, transform=transform_train(args.dataset),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate, random_seed=args.seed,
                                              noise_type=args.noise_type, num_class=args.num_class)
        val_data = dataloader.mnist_dataset(False, dir=args.data_dir, transform=transform_test(args.dataset),
                                            target_transform=transform_target,
                                            noise_rate=args.noise_rate, random_seed=args.seed,
                                            noise_type=args.noise_type, num_class=args.num_class)
        test_data = dataloader.mnist_test_dataset(dir=args.data_dir, transform=transform_test(args.dataset),
                                                  target_transform=transform_target)

    if args.dataset == 'cifar10':
        train_data = dataloader.cifar10_dataset(True, dir=args.data_dir, transform=transform_train(args.dataset),
                                                target_transform=transform_target,
                                                noise_rate=args.noise_rate, random_seed=args.seed,
                                                noise_type=args.noise_type, num_class=args.num_class)
        val_data = dataloader.cifar10_dataset(False, dir=args.data_dir, transform=transform_test(args.dataset),
                                              target_transform=transform_target,
                                              noise_rate=args.noise_rate, random_seed=args.seed,
                                              noise_type=args.noise_type, num_class=args.num_class)
        test_data = dataloader.cifar10_test_dataset(dir=args.data_dir, transform=transform_test(args.dataset),
                                                    target_transform=transform_target)

    return train_data, val_data, test_data


def prepare_task_target(target):
    train_target = Class2Simi(target, mode='hinge')
    eval_target = target
    return train_target.detach(), eval_target.detach()  # Make sure no gradients


def estimate(n_epoch, train_loader, estimate_loader, val_loader, learner, train_data, True_T):
    # prob save files
    index_num = int(len(train_data) / args.batch_size)
    A = torch.zeros((n_epoch, len(train_data), args.num_class))
    val_acc_list = []

    # path for prob files and matrix
    prob_save_dir = args.output_dir + args.prob_dir + '/' + str(args.seed)
    if not os.path.exists(prob_save_dir):
        os.system('mkdir -p %s' % (prob_save_dir))

    matrix_save_dir = args.output_dir + args.matrix_dir + '/' + str(args.seed)
    if not os.path.exists(matrix_save_dir):
        os.system('mkdir -p %s' % (matrix_save_dir))

    model_save_dir = args.output_dir + 'model'
    if not os.path.exists(model_save_dir):
        os.system('mkdir -p %s' % (model_save_dir))

    # The optimization loop
    q = np.eye(args.num_class)
    q = torch.from_numpy(q).float().cuda()

    for epoch in range(n_epoch):
        print('epoch {}'.format(epoch))
        # training-----------------------------
        train_loss = 0.
        for i, (input, target) in enumerate(train_loader):
            learner.train()
            # Prepare the inputs
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            loss, output = learner.learn_class(input, target, q)  # output is already prob.
            torch.save(learner.model.state_dict(), model_save_dir + '/' + 'epoch_%d.pth' % (epoch))
            train_loss += loss
        learner.est_step_schedule(epoch)
        # val and decide the location of stopping for the choice of test accuracy
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        for i, (input, target) in enumerate(val_loader):
            learner.eval()

            # Prepare the inputs
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # Optimization
            output = learner.model(input)
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().float()
            total += len(target)
            output = output.detach()

            # Loss-specific information
        acc = (correct / total).detach()
        # print('[Validation] ACC: ', acc.item())
        val_acc_list.append(acc.item())
        correct = torch.zeros(1).squeeze().cuda()
        total = torch.zeros(1).squeeze().cuda()
        for i, (input, target) in enumerate(estimate_loader):
            learner.eval()
            # Prepare the inputs
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # Optimization
            loss, output = learner.learn_class_val(input, target, q)
            prediction = torch.argmax(output, 1)
            correct += (prediction == target).sum().float()
            total += len(target)
            output = output.detach()

            # Loss-specific information
            if i <= index_num:
                A[epoch][i * args.batch_size:(i + 1) * args.batch_size, :] = output
            else:
                A[epoch][index_num * args.batch_size, len(train_data), :] = output

        acc = (correct / total).detach()
        print("[Train] ACC: ", acc.item())

    val_acc_array = np.array(val_acc_list)
    model_index = np.argmax(val_acc_array)

    A_save_dir = prob_save_dir + '/' + 'prob.npy'
    np.save(A_save_dir, A)

    prob = np.load(A_save_dir)
    transition_matrix = tools.fit(prob[model_index, :, :], args.num_class,
                                  per_radio=args.anchorrate)
    transition_matrix = tools.norm(transition_matrix)
    matrix_path = matrix_save_dir + '/' + 'transition_matrix_%s.npy' % (args.seed)
    np.save(matrix_path, transition_matrix)

    print("\nClass transition matrix: \n", transition_matrix)
    simi_T = tools.class2simi(transition_matrix)
    print("\nSimilarity transition matrix: \n", simi_T)
    T = torch.from_numpy(simi_T).float()
    return T, model_index


def train(epoch, train_loader, learner, q, args):

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    print('\n==== Epoch:{0} ===='.format(epoch))
    learner.train()

    for i, (input, target) in enumerate(train_loader):

        # Prepare the inputs
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        train_target, eval_target = prepare_task_target(target)

        loss, output = learner.learn(input, train_target, q)

        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().float()
        total += len(target)
        output = output.detach()

    learner.step_schedule(epoch, args.dataset)

    acc = (correct / total).detach()
    print('[Train] Loss: ', loss.item())
    print('[Train] ACC: ', acc.item())

    return loss.item(), acc.item()


def evaluate(eval_loader, model, type):
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()
    model.eval()
    for i, (input, target) in enumerate(eval_loader):

        if torch.cuda.is_available():
            with torch.no_grad():
                input = input.cuda()
                target = target.cuda()

        output = model(input)
        prediction = torch.argmax(output, 1)
        correct += (prediction == target).sum().float()
        total += len(target)
        output = output.detach()

    acc = (correct / total).detach()
    if type == 'val':
        print('\n---- Validation ----')
        print('[Validation] ACC: ', acc.item())
    if type == 'test':
        print('\n---- Evaluation ----')
        print('[Test] ACC: ', acc.item())

    return acc.item()


def main(args):
    if args.dataset == 'cifar10':
        args.num_class = 10
        args.net = 'resnet'
        args.batch_size = 512
        args.estimate_epochs = 30
        args.epochs = 200
        args.anchorrate = 100
        args.weight_decay = 1e-5

    if args.dataset == 'mnist':
        args.num_class = 10
        args.net = 'lenet'
        args.batch_size = 128
        args.estimate_epochs = 8
        args.epochs = 30
        args.anchorrate = 90
        args.weight_decay = 1e-4

    if args.loss == 'forward':
        criterion = forward_MCL()
    if args.loss == 'reweight':
        criterion = reweight_MCL()

    if args.net == 'lenet':
        model = Lenet.LeNet()
        model_train = Lenet.LeNet()
    if args.net == 'resnet':
        model = ResNet26.ResNet26(args.num_class)
        model_train = ResNet26.ResNet26(args.num_class)

    print('Parameter:', args)

    print('\nLoading data ...')

    train_data, val_data, test_data = loaddata(args)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=False)

    estimate_loader = DataLoader(dataset=train_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 shuffle=False,
                                 drop_last=False)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False,
                            drop_last=False)

    eval_loader = DataLoader(dataset=test_data,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False,
                             drop_last=False)

    if torch.cuda.is_available():
        model = model.cuda()
        model_train = model_train.cuda()
        criterion = criterion.cuda()

    optim_args = {'lr': args.lr, 'weight_decay': args.weight_decay}
    optim_es_args = {'lr': args.lr_es, 'weight_decay': args.weight_decay}

    optimizer_estimate = torch.optim.__dict__[args.optimizer_es](model.parameters(), **optim_es_args)
    scheduler_estimate = StepLR(optimizer_estimate, step_size=5, gamma=1)
    learner_estimate = Learner_Likelihood(model, criterion, optimizer_estimate, scheduler_estimate)

    optimizer = torch.optim.__dict__[args.optimizer](model_train.parameters(), **optim_args)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    learner = Learner_Likelihood(model_train, criterion, optimizer, scheduler)

    noise_type = args.noise_type

    if noise_type == 'none':
        t = np.eye(args.num_class)
        simi_T = tools.class2simi(t)
        q, index = estimate(args.estimate_epochs, train_loader, estimate_loader, val_loader, learner_estimate,
                            train_data, t)
        print('\n\n==== Load %d model ====' % index)
        model_train.load_state_dict(torch.load(args.output_dir + 'model' + '/' + 'epoch_' + str(index) + '.pth'))

    if noise_type == 's':
        t = tools.s_transition_matrix_generate(noise_rate=args.noise_rate, num_classes=args.num_class)
        print('\n==== Estimating transition matrix ====')
        q, index = estimate(args.estimate_epochs, train_loader, estimate_loader, val_loader, learner_estimate,
                            train_data, t)
        print('\n==== Load %d model ====' % index)
        model_train.load_state_dict(torch.load(args.output_dir + 'model' + '/' + 'epoch_' + str(index) + '.pth'))

    if noise_type == 'as':
        t = tools.rand_transition_matrix_generate(noise_rate=args.noise_rate, num_classes=args.num_class)
        print('\n==== Estimating transition matrix ====')
        q, index = estimate(args.estimate_epochs, train_loader, estimate_loader, val_loader, learner_estimate,
                            train_data, t)
        print('\n==== Load %d model ====' % index)
        model_train.load_state_dict(torch.load(args.output_dir + 'model' + '/' + 'epoch_' + str(index) + '.pth'))

    if torch.cuda.is_available():
        q = q.cuda()

    train_model_save_dir = args.output_dir + 'train_model'
    if not os.path.exists(train_model_save_dir):
        os.system('mkdir -p %s' % train_model_save_dir)

    train_acc_list = []
    val_acc_list = []
    acc_list = []

    print("\nTraining with %s" % args.loss)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = train(epoch, train_loader, learner, q, args)
        train_acc_list.append(tr_acc)
        val_acc = evaluate(val_loader, model_train, 'val')
        val_acc_list.append(val_acc)
        accuracy = evaluate(eval_loader, model_train, 'test')
        acc_list.append(accuracy)
        torch.save(learner.model.state_dict(), train_model_save_dir + '/' + 'epoch_%d.pth' % epoch)

    index = np.argmax(np.array(val_acc_list))
    max_index = np.argmax(np.array(acc_list))
    print('\nvalidation acc max epoch', index, 'acc:', acc_list[index])
    print('final epoch acc:', acc_list[-1], '\nmax acc', acc_list[max_index])

    return acc_list[index], acc_list[-1], acc_list[max_index]


if __name__ == "__main__":

    # Run from noise rate [args.r, args.r + args.bar] with 0.1 interval
    for m in np.arange(args.r, args.r + args.bar, 0.1):
        report_list = []
        args.noise_rate = m

        # Run args.n seeds
        for i in range(args.n):
            args.seed = i + 1
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            np.random.seed(args.seed)
            args.output_dir = './' + args.d + '/' + str(args.noise_rate) + '/'
            if not os.path.exists(args.output_dir):
                os.system('mkdir -p %s' % (args.output_dir))
            # Print in terminal or file
            if args.p == 0:
                f = open(args.output_dir + 'simi_' + str(args.noise_type) + '_' + str(args.dataset) + '_' + str(
                    args.seed) + '.txt', 'a')
                sys.stdout = f
                sys.stderr = f

            output_acc, _, _ = main(args)
            report_list.append(output_acc)

        if args.loss == 'forward':
            print('\nforward_acc:', report_list)
            print('\nforward_acc', 'mean:', np.array(report_list).mean(), 'std:',
                  np.array(report_list).std(ddof=1))
        else:
            print('\nreweight_acc:', report_list)
            print('\nreweight_acc', 'mean:', np.array(report_list).mean(), 'std:',
                  np.array(report_list).std(ddof=1))
