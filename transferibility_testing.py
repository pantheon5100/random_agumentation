
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3

# Untrained model
my_model = vgg11_bn()

# Pretrained model
my_model = vgg11_bn(pretrained=True)
my_model.eval() # for evaluation




import torch
import torch.nn.functional as F
from preact_resnet import PreActResNet18
import torchattacks
from torchvision import transforms, datasets



import torchattacks

# general 
# log_files/supervised_baseline_eps8/supervised_baseline_eps8-image_normalize--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_0-seed_0/20220113083538/model.pth
# noise aug trained
# log_files/supervised_noise_aug_eps8/supervised_noise_aug_eps8-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_0-seed_0/20220113083721/model.pth


general_trained = "../NoiseAug/log_files/supervised_baseline_eps8/supervised_baseline_eps8-image_normalize--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_0-seed_0/20220113083538/model.pth"
noise_aug_trained = "../NoiseAug/log_files/supervised_noise_aug_eps8/supervised_noise_aug_eps8-image_normalize--NoiseAug-_type_uniform-noise_aug_size_2.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_0-seed_0/20220113083721/model.pth"

pgd2_noise_aug = "../NoiseAug/log_files/pgd2-noise-aug/pgd2-noise-aug-image_normalize--NoiseAug-_type_normal-noise_aug_size_1.0--epochs_30-lr_schedule_cyclic-lr_max_0.3-epsilon_8-attack_steps_2-alpha_4.0-delta_init_zero-zero_one_clamp_1-seed_0/20220114025153/model.pth"

state_dict = torch.load(general_trained)
model_general_trained = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
model_general_trained.load_state_dict(state_dict)


state_dict = torch.load(noise_aug_trained)
model_noiseaug_trained = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
model_noiseaug_trained.load_state_dict(state_dict)

state_dict = torch.load(pgd2_noise_aug)
pgd2_noise_aug = PreActResNet18().cuda() ### to make sure that the robustness evaluation is done on single precision instead of half-precision
pgd2_noise_aug.load_state_dict(state_dict)


dir_ = "/dev/shm"
batch_size = 128
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar10_mean, cifar10_std),
])

test_dataset = datasets.CIFAR10(
    dir_, train=False, transform=test_transform, download=True)
test_dataset.data = test_dataset.data[:1000]
test_dataset.targets = test_dataset.targets[:1000]


test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=2,
)



# transfer testing model
target_model_list = {}

target_model = vgg11_bn(pretrained=True).cuda()
target_model.eval() # for evaluation
target_model_list["vgg11_bn"] = target_model


target_model = vgg13_bn(pretrained=True).cuda()
target_model.eval() # for evaluation
target_model_list["vgg13_bn"] = target_model

# target_model = vgg16_bn(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["vgg16_bn"] = target_model

# target_model = vgg19_bn(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["vgg19_bn"] = target_model

# target_model = resnet18(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["resnet18"] = target_model

# target_model = resnet34(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["resnet34"] = target_model

# target_model = resnet50(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["resnet50"] = target_model

# target_model = densenet121(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["densenet121"] = target_model

# target_model = densenet161(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["densenet161"] = target_model

# target_model = densenet169(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["densenet169"] = target_model

# target_model = mobilenet_v2(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["mobilenet_v2"] = target_model

# target_model = googlenet(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["googlenet"] = target_model

# target_model = inception_v3(pretrained=True).cuda()
# target_model.eval() # for evaluation
# target_model_list["inception_v3"] = target_model




def transfer_eval(source_model):

    # attack images generator
    atk = torchattacks.PGD(source_model, eps=16*4/255, alpha=2*4/255, steps=10)


    test_loss = 0
    test_acc = 0
    n = 0
    source_model.eval()

    transferibility_recording = {model_name:[0, 0, 0] for model_name, _ in target_model_list.items()}
    transferibility_recording["source_model"] = [0, 0, 0]

    # with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        n += y.size(0)

        # generate adv_images
        adv_images = atk(X, y)

        output = source_model(adv_images)
        correct = (output.max(1)[1] == y).sum().item()

        transferibility_recording["source_model"][0] += correct
        transferibility_recording["source_model"][1] = transferibility_recording["source_model"][0]/n


        for model_name, model_target in target_model_list.items():
            output = model_target(adv_images)
            correct = (output.max(1)[1] == y).sum().item()

            transferibility_recording[model_name][0] += correct
            transferibility_recording[model_name][1] = transferibility_recording[model_name][0]/n
        
        # print(f"batch {i}")

    print(transferibility_recording)



def clean_test():

    n = 0
    transferibility_recording = {model_name:[0, 0, 0] for model_name, _ in target_model_list.items()}

    # with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        X, y = X.cuda(), y.cuda()
        n += y.size(0)

        for model_name, model_target in target_model_list.items():
            output = model_target(X)
            correct = (output.max(1)[1] == y).sum().item()

            transferibility_recording[model_name][0] += correct
            transferibility_recording[model_name][1] = transferibility_recording[model_name][0]/n
        
        # print(f"batch {i}")

    print(transferibility_recording)

# print("model_general_trained")
# transfer_eval(model_general_trained)
# print("model_noiseaug_trained")

# transfer_eval(model_noiseaug_trained)

# print("pgd2_noise_aug")

# transfer_eval(pgd2_noise_aug)

# resnet50_model = resnet50(pretrained=True).cuda()
# resnet50_model.eval() # for evaluation
# print("resnet50_model")

# transfer_eval(resnet50_model)
# print("clean_test")

# clean_test()




'''
General trained model

{'vgg11_bn': [5048, 0.5048, 0], 'vgg13_bn': [4018, 0.4018, 0], 'vgg16_bn': [4092, 0.4092, 0], 'vgg19_bn': [4257, 0.4257, 0], 'resnet1
8': [4644, 0.4644, 0], 'resnet34': [4743, 0.4743, 0], 'resnet50': [4616, 0.4616, 0], 'densenet121': [4698, 0.4698, 0], 'densenet161':
 [5257, 0.5257, 0], 'densenet169': [4750, 0.475, 0], 'mobilenet_v2': [3209, 0.3209, 0], 'googlenet': [3022, 0.3022, 0], 'inception_v3
': [3315, 0.3315, 0], 'General': [831, 0.0831, 0]}

NoiseAug trained model

{'vgg11_bn': [5513, 0.0006, 0], 'vgg13_bn': [5260, 0.0006, 0], 'vgg16_bn': [5308, 0.0006, 0], 'vgg19_bn': [5415, 0.0007, 0], 'resnet1
8': [5283, 0.0005, 0], 'resnet34': [5396, 0.0007, 0], 'resnet50': [5318, 0.0006, 0], 'densenet121': [5376, 0.0005, 0], 'densenet161':
 [5872, 0.0008, 0], 'densenet169': [5407, 0.0006, 0], 'mobilenet_v2': [5001, 0.0005, 0], 'googlenet': [4152, 0.0004, 0], 'inception_v
3': [4732, 0.0007, 0], 'NoiseAug': [337, 0.0001, 0]}


'''