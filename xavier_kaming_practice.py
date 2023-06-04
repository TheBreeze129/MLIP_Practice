from VGGNet_practice import *

def xavier_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

def kaiming_init(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

model_x = ResNet(BottleNeck, [3,4,6,3], num_classes=10)
model_k = ResNet(BottleNeck, [3,4,6,3], num_classes=10)

model_x.apply(xavier_init)
model_k.apply(kaiming_init)

criterion_x = nn.CrossEntropyLoss()
criterion_k = nn.CrossEntropyLoss()

optimizer_x = torch.optim.Adam(model_x.parameters(), lr=learning_rate)
optimizer_k = torch.optim.Adam(model_k.parameters(), lr=learning_rate)

model_x.to(device)
model_k.to(device)

# Xavier
print("Xavier Normal")
num_epochs = 10
model_x.train()
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer_x.zero_grad()
        outputs = model_x(images)
        loss = criterion_x(outputs, labels)
        loss.backward()
        optimizer_x.step()

    # test
    model_x.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_x(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}%")

# Kaiming
print("Kaming Normal")
model_k.train()
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer_k.zero_grad()
        outputs = model_k(images)
        loss = criterion_k(outputs, labels)
        loss.backward()
        optimizer_k.step()

    #test
    model_k.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_k(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.4f}%")

