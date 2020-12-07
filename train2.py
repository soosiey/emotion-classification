def train(epoch, num_classes, bins, class_=0):
  model.train()
  cnt = 0
  train_total = 0
  train_correct = 0
  total_loss = 0
  for batch in (train_loader):
    data = batch['data'].to(device).double()
    if num_classes == 1:
      labels = batch['labels'][:,class_,np.newaxis].to(device).double()
    else:
      labels = batch['labels'].to(device).double()
    labels = labels[:,:3]
    digitized_labels = torch.bucketize(labels, bins)
    output = model(data)
    preds = torch.max(output,1)[1]
    print(preds)
    if(epoch > 15):
      print(output)
    total_loss += labels.size(0)
    loss = criterion(output, digitized_labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    cnt += 1
    
    preds = torch.bucketize(output, bins)
    # print(preds)
    train_correct += (preds == digitized_labels).all(1).sum().item()
    train_total += labels.size(0)

  if(epoch % 1 == 0):
    print('----------------------------------------')
    print('Epoch',epoch)
    print('Average Loss:', total_loss/cnt)
    print('Train Accuracy:', train_correct, "/", train_total, ":", train_correct/train_total)

def test(epoch, num_classes, bins, class_=0):
  model.eval()
  cnt = 0
  total_loss = 0
  test_total = 0
  test_correct = 0
  for tbatch in (test_loader):
      data = tbatch['data'].to(device).double()
      if num_classes == 1:
        labels = tbatch['labels'][:,class_,np.newaxis].to(device).double()
      else:
        labels = tbatch['labels'].to(device).double()
      labels = labels[:,:3]
      output = model(data)
      total_loss += criterion(output, labels).item()
      cnt += 1

      preds = torch.bucketize(output, bins)
      digitized_labels = torch.bucketize(labels, bins)
      test_correct += (preds == digitized_labels).all(1).sum().item()
      test_total += labels.size(0)

  print('Testing Epoch', epoch)
  print('Testing Loss:', total_loss/cnt)
  print('Testing Accuracy:', test_correct, "/", test_total, ":", test_correct/test_total)


if __name__ == '__main__':
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = EmotionDataset.EmotionDataset()
    test_dataset = EmotionDataset.EmotionDataset(train=False)

    batch_size=64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    num_layers = 4
    num_hidden = 100
    num_classes = 3
    CUDA = True

    model = SpectrogramLSTM(num_layers, num_hidden, num_classes)
    device = 'cuda' if CUDA and torch.cuda.is_available() else 'cpu'
    model.double().to(device)

    print(device)

    lr = 1e-8
    epochs = 100
    weight_decay = 0.01

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = lr)#, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,15,30,50],gamma=.1)
    # optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay=weight_decay)

    bins = torch.linspace(1, 9, 4).to(device)
    # bins = torch.linspace(1, 9, 3).to(device)
    bins[-1] += 0.1
    bins[0] -= 0.1

    class_ = 0
    for epoch in range(100):
      train(epoch, num_classes, bins, class_)
      test(epoch, num_classes, bins, class_)
      scheduler.step()