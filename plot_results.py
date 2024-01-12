import pandas as pd
import matplotlib.pyplot as plt

# # Generate predictions
# train_cnn.test_generator.reset()

# preds = train_cnn.best_model.predict(
#     train_cnn.test_generator,
#     verbose = 1
# )

# test_results = pd.DataFrame({
#     "Filename": train_cnn.test_generator.filenames,
#     "Prediction": preds.flatten()
# })
# print(test_results)
filepath = r"results.csv"
   
# read the CSV file 
df = pd.read_csv(filepath) 
   
# print the first five rows 
print(df.head()) 

acc = df['accuracy']
val_acc = df['val_accuracy']
loss = df['loss']
val_loss = df['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label = 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()