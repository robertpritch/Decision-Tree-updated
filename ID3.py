from node import Node
import math

def ID3(examples, default):
  '''
  Takes in an array of examples, and returns a tree (an instance of Node) 
  trained on the examples.  Each example is a dictionary of attribute:value pairs,
  and the target class variable is a special attribute with the name "Class".
  Any missing attributes are denoted with a value of "?"
  '''
  
  class_data = class_options(examples)
  
  # If there's only one class in the data, create a leaf node with that class label
  if len(class_data[0]) == 1:
    tree_node = Node()
    tree_node.label = class_data[0][0]
    return tree_node
  
  # Finding the mode (most common class) for later use
  temp_max = max(class_data[1])
  
  i_max = class_data[1].index(temp_max)
  
  mode_d = class_data[0][i_max]
  
  tree_node = Node()
  tree_node.label = mode_d
  
  # If there are no attributes left to split on, return a leaf node with the mode class
  if len(examples[0]) == 1:
    
    return tree_node
  else:
    # Selecting the best attribute based on information gain
    attr = info_gain(examples, class_data, mode_d)
    
    new_data = {}

    #default stores all the data, so that always have access to it even after you remove certain attributes on branches
    #checks if default is not a dictionary, then this is the first iteration of ID3 and it stores examples
    if not isinstance(default, dict):
      default = examples
    
    # Organizing data for the next level of recursion
    all_params = list_params(attr, default)
    
    #new data will hold the examples with the best attribute removed
    for param in all_params:
      new_data[param] = []
    for example in examples:
      temp_ex = dict(example)
      save_data = temp_ex.pop(attr)
      new_data[save_data].append(temp_ex)
      
    # Recursively building the subtree for each value of the best attribute
    for key in new_data:

      #if there are no datapoints in examples that corellate to this paramter, then just label the leaf with the most common target/class. Ex. slide 37 on Lecture 04 slides
      if len(new_data[key]) == 0 :
        
        leaf = Node()
        leaf.label = mode_d
        tree_node.children["Class"] = leaf

      #otherwise label this branch node with the best attr and run ID3 with the attribute removed
      else:
        tree_node.label = attr
        tree_node.children[key] = ID3(new_data[key], default)
        
  
  return tree_node  # Returning the constructed tree

def prune(node, examples):
    '''
    Takes in a trained tree and a validation set of examples. Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    '''
    original_accuracy = test(node, examples)
    
    for value, child in node.children.items():
        prune(child, examples)  # Recursive pruning

        # Temporarily prune the node
        majority_label = get_majority_label(child)  # Get the majority class from the subtree
        temp_node = Node()
        temp_node.label = majority_label

        # Replace the current child with a leaf node containing the majority class
        original_child = child
        node.children[value] = temp_node

        # Get the accuracy after pruning
        pruned_accuracy = test(node, examples)

        # If the accuracy does not improve, restore the original child
        if original_accuracy >= pruned_accuracy:
            node.children[value] = original_child

def test(node, examples):
  '''
  Takes in a trained tree and a test set of examples.  Returns the accuracy (fraction
  of examples the tree classifies correctly).
  '''
  correct = 0  # Counter for correct predictions
    
  # Iterate through each example
  for example in examples:
      prediction = evaluate(node, example)  # Get the predicted class
      if prediction == example['Class']:  # Check if the prediction is correct
          correct += 1
  
  # Calculate accuracy
  accuracy = correct / len(examples)
  
  return accuracy
  


def evaluate(node, example):
  '''
  Takes in a tree and one example.  Returns the Class value that the tree
  assigns to the example.

  '''
  label = node.label
  if len(node.children) == 0:
      return label
  elif example[label] not in node.children or example[label] == '?':  # Handle unseen or missing values
      return get_majority_class([example])  # Assumes a function to find the majority class
  else:
      return evaluate(node.children[example[label]], example)
  

#cycles through each attribute, makes a dictionary which stores the attribute options and the classes (like the table on Slide 34 on the Lecture 04 slides)

def info_gain(examples, classes, mode_d):
  
  first_example = examples[0]
  attribute_list = list(first_example.keys())

  info_best_name = ""
  info_best_val = 0
  classes = classes[0]

  # Determine the majority class from the examples
  majority_class = get_majority_class(examples)

  for attribute in attribute_list:
    total = 0
    param_count = {}
    if attribute != "Class":
      for example in examples:
        if example[attribute] not in param_count:
          class_response = classes.index(example["Class"])
          param_count[example[attribute]] = [0] * len(classes)
          # If missing attribute, use majority class
          if example[attribute] == '?':
            class_response = classes.index(majority_class)
          param_count[example[attribute]][class_response] = 1
          total += 1
        else:
          class_response = classes.index(example["Class"])
          # If missing attribute, use majority class
          if example[attribute] == '?':
            class_response = classes.index(majority_class)
          param_count[example[attribute]][class_response] += 1
          total += 1

      temp_gain = con_entropy(param_count, total)
      if info_best_name == "":
        info_best_name = attribute
        info_best_val = temp_gain
      elif info_best_val > temp_gain:
        info_best_val = temp_gain
        info_best_name = attribute
  
  return info_best_name

    
#This calculates the entropy for a given attribute. It just uses the formula given in class
def con_entropy(param_count, total):
  entrop = 0
  
  for param in param_count:
    prob = 0
    prob_mult = 0
    for i in range(len(param_count[param])):
      prob += param_count[param][i]
      if param_count[param][i] != 0:
        prob_mult += param_count[param][i] / total * math.log(param_count[param][i] / total, 2)
    
    entrop += (prob / total) * prob_mult
  return -1 * entrop
    


#Returns a list of two lists [class_response, class_count]. Class response is a list that holds all the possible targets (ex. ["democrat", "republican"]) in house_votes_84.data
#class_count holds the number of examples that correspond to each class response. Ex. [["democrat, repbulican"], [40, 50]] means that there were 40 examples with Target democrat.
        
def class_options(examples):
  class_response = []
  class_count = []
  for example in examples:
    if example["Class"] not in class_response:
      class_response.append(example["Class"])
      class_count.append(1)
    else:
      class_count[class_response.index(example["Class"])] += 1
  return [class_response, class_count]


#returns a list of the possible options for a given attribute. Ex. calling this function on house_votes_84.data attribute: handicapped-infants would return ["y", "n"]
def list_params(attr, examples):
  param_list = []
  for example in examples:
    if example[attr] not in param_list:
      param_list.append(example[attr])
  return param_list

def collect_labels(node):
    '''
    Recursively collect all labels from the leaf nodes in a subtree.
    '''
    labels = []
    if len(node.children) == 0:
        labels.append(node.label)
    else:
        for child in node.children.values():
            labels.extend(collect_labels(child))
    return labels

def get_majority_label(node):
    '''
    Get the majority label in a subtree.
    '''
    labels = collect_labels(node)
    class_counts = {}  # Dictionary to hold class counts
    for label in labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Find and return the class with the maximum count
    majority_label = max(class_counts, key=class_counts.get)
    return majority_label

def get_majority_class(examples):
    class_counts = {}  # Dictionary to hold class counts
    for example in examples:
        class_label = example['Class']
        class_counts[class_label] = class_counts.get(class_label, 0) + 1
    
    # Find and return the class with the maximum count
    majority_class = max(class_counts, key=class_counts.get)
    return majority_class