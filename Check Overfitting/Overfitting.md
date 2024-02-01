## Train the model
svm.fit(X_train, y_train)

## Obtain predictions on both training and test sets
train_pred = svm.predict(X_train)
test_pred = svm.predict(X_test)

## Calculate accuracy scores for training and test sets
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

## Print the scores
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
