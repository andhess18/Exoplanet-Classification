import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from preprocessing import preprocess_data
from model import NeuralNet
from train import train
from train import calculate_accuracy
from preprocessing import preprocess_data

def main():
    X_train, X_test, y_train, y_test = preprocess_data('exoplanet_data.csv')

    # Convert data to PyTorch Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    
    #initializes Data Loaders
    batch_size = 64 #number of samples propagated through the network at once
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    #initializes NeuralNetwork and trains it
    input_dim = X_train.shape[1]
    model = NeuralNet(input_dim)
    train(model, train_loader, test_loader, epochs=50, lr=0.001)
    
    accuracy = calculate_accuracy(model, test_loader)
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')


    #saves model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()