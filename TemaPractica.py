import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def load_data(folder_path):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.startswith("spmsg"):
            # 1 indicates spam
            labels.append(1)
        else:
            # 0 indicates non-spam
            labels.append(0)

        with open(file_path, "r", encoding="latin-1") as f:
            content = f.read()
            data.append(content)

    return data, labels

def process_files(main_folder_path):
    folders = [f"part{i}" for i in range(1, 11)]
    for folder in folders:
     folder_path = os.path.join(main_folder_path, folder)
 
     # Load data from the specified folder
     data, labels = load_data(folder_path)
 
     if folder == "part10":
         # For part10, use the data only for testing, not for training
         vectorizer = CountVectorizer(vocabulary=vectorizer.vocabulary_)
         X_test = vectorizer.transform(data)
         y_test = labels
     else:
         # For other folders, use the data for both training and testing
         vectorizer = CountVectorizer()
         X = vectorizer.fit_transform(data)
         X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
 
         # Create and train the Naive Bayes classifier
         nb_classifier = MultinomialNB()
         nb_classifier.fit(X_train, y_train)
 
     # Make predictions on the test set
     y_pred = nb_classifier.predict(X_test)
 
     # Evaluate the performance
     accuracy = accuracy_score(y_test, y_pred)
     print(f'\nResults for folder {folder}:')
     print(f'Accuracy: {accuracy:.2f}')
 
     # Print classification report and confusion matrix
     print('\nClassification Report:')
     print(classification_report(y_test, y_pred))
 
     print('\nConfusion Matrix:')
     conf_matrix = confusion_matrix(y_test, y_pred)
     print(conf_matrix)


def leave_one_out_cross_validation(main_folder_path, subdirectories):
    accuracies = []

    for subdirectory in subdirectories:
        folder_path = os.path.join(main_folder_path, subdirectory)
        folders = [f"part{i}" for i in range(1, 11)]

        for folder in folders:
            folder_path = os.path.join(main_folder_path, subdirectory, folder)

            # Load data for training
            train_data = []
            train_labels = []
            for train_folder in folders:
                if train_folder != folder:
                    train_folder_path = os.path.join(main_folder_path, subdirectory, train_folder)
                    data, labels = load_data(train_folder_path)
                    train_data.extend(data)
                    train_labels.extend(labels)

            # Load data for testing
            test_folder_path = os.path.join(main_folder_path, subdirectory, folder)
            test_data, test_labels = load_data(test_folder_path)

            # Vectorize the data
            vectorizer = CountVectorizer()
            X_train = vectorizer.fit_transform(train_data)
            X_test = vectorizer.transform(test_data)

            # Create and train the Naive Bayes classifier
            nb_classifier = MultinomialNB()
            nb_classifier.fit(X_train, train_labels)

            # Make predictions on the test set
            y_pred = nb_classifier.predict(X_test)

            # Evaluate accuracy
            accuracy = accuracy_score(test_labels, y_pred)
            accuracies.append(accuracy)

    # Calculate and print average accuracy
    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f'\nAverage Accuracy across Leave-One-Out Cross-Validation: {avg_accuracy:.2f}')

    return accuracies

# Specify the path to the folder containing messages
main_folder_path = "lingspam_public"
subdirectories = ["bare", "lemm", "lemm_stop", "stop"]

# Get accuracies for Leave-One-Out Cross-Validation
accuracies = leave_one_out_cross_validation(main_folder_path, subdirectories)

# Plot the accuracies
plt.figure(figsize=(10, 6))
plt.plot(accuracies, marker='o')
plt.title('Leave-One-Out Cross-Validation Accuracies')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
