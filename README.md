# mini_project
1. I hope you download all the repository into VScode cause I put all necessary files in it.
2. mini_project.py is the main project, just run it.
3. model_visualizaion.pdf is the whole Resnet model structure image which is partly shown in report.
4. In mini_project.py, about trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transform_train), just as I have already put the cifar-10 dataset into this repository. If you just download only one mini_project.py, you need change the code into root='./data' and download=True. Testset is the same.
5. The mini_project.ipynb just for shown, showing all the code structure and all the outputs(including the number of all parameters).
6. If you lack some packages, you can use the requirements.txt to install all necessary packages.
7. The project.pth file is just for testing. As all the diagrams(such as the learning rate chart and the loss chart) are made by the training process. If I just use the project.pth to test, it just output the csv file and the accuracy number. If you just want to use the project.pth, you can just comment on the training part of the code and add 1.checkpoint = torch.load('project.pth') 2.net.load_state_dict(checkpoint['state_dict']) then the model weight is loaded.
