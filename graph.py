import matplotlib.pyplot as plt
import random
import ID3  # Assuming your ID3 code is in a file named ID3.py
import parse  # Assuming you have a parse function to read the data

def load_data(file_path):
    # Assuming you have a function to load and parse your data
    return parse.parse(file_path)

def split_data(data, train_size):
    random.shuffle(data)
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def main():
    data = load_data('house_votes_84.data')
    training_sizes = range(10, 301, 10)  # Adjust the step as needed
    accuracies_with_pruning = []
    accuracies_without_pruning = []

    for size in training_sizes:
        acc_with_pruning = 0
        acc_without_pruning = 0
        for _ in range(100):
            train, test = split_data(data, size)
            
            # Without Pruning
            tree = ID3.ID3(train, 'democrat')  
            acc_without_pruning += ID3.test(tree, test)
            
            # With Pruning
            ID3.prune(tree, train)  # Assuming your prune function uses the training data for pruning
            acc_with_pruning += ID3.test(tree, test)
        
        accuracies_with_pruning.append(acc_with_pruning / 100)
        accuracies_without_pruning.append(acc_without_pruning / 100)

    # Plotting
    plt.plot(training_sizes, accuracies_with_pruning, label='With Pruning')
    plt.plot(training_sizes, accuracies_without_pruning, label='Without Pruning')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy on Test Data')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()