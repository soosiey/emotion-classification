import torch
import torch.nn as nn
import EmotionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from model import CustomNet
from tqdm import tqdm
import logging

def main():

    logging.basicConfig(level=logging.DEBUG, filename='model4.log', filemode='a+', format='%(asctime) - 15s %(levelname) - 8s %(message) s')
    batch_size = 8
    lr = .01
    epochs = 101
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = EmotionDataset.EmotionDataset()
    test_dataset = EmotionDataset.EmotionDataset(train=False)

    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CustomNet(device=device, n_layers=10, h_size=20).double()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr = lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,25,40,80], gamma=.1)

    logging.info('Using ' + device)
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        cnt = 0
        for batch in tqdm(train_loader):
            data = batch['data'].to(device)
            labels = batch['labels'].to(device)

            output = model(data)
            preds = torch.max(output,1)[1]
            correct += (preds == labels).all(1).sum().item()
            total += labels.size(0)
            loss = criterion(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1
        logging.info('----------------------------------------')
        logging.info('Epoch ' + str(epoch))
        logging.info('Average Loss: ' + str(total_loss/cnt))
        logging.info('Accuracy: ' + str(correct / total))
        if(epoch % 10 == 0):
            model.eval()
            test_correct = 0
            test_total = 0
            for tbatch in tqdm(test_loader):
                data = tbatch['data'].to(device)
                labels = tbatch['labels'].to(device)

                output = model(data)
                preds = torch.max(output,1)[1]
                test_correct += (preds == labels).all(1).sum().item()
                test_total += labels.size(0)
            logging.info('Testing Epoch ' + str(epoch))
            logging.info('Testing Accuracy: ' + str(test_correct/test_total))
        scheduler.step()
        if(epoch % 20 == 0):
            torch.save(model.state_dict(), 'trained_models/model4/epoch_'+str(epoch)+'.model')




if __name__ == '__main__':
    main()