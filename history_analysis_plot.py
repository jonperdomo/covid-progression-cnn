import numpy as np
from matplotlib import pyplot as plt

history_path = 'Models/Model_4_history.npy'
h = np.load(history_path, allow_pickle=True).item()

# # Training accuracy
# acc = h['accuracy']
# for i in range(len(acc)):
#     # if float(acc[i]) == 1:
#     #     print(i)
#     print(acc[i])

# # Validation accuracy
# acc = h['val_accuracy']
# acc_diff = np.diff(acc)
# for i in range(len(acc_diff)):
#     print("%d, Diff: %.3f, Val: %.3f" % (i, acc_diff[i], acc[i]))

# # Loss
# acc = h['loss']
# acc_diff = np.diff(acc)
# for i in range(len(acc_diff)):
#     print("%d, Diff: %.3f, Val: %.6f" % (i, acc_diff[i], acc[i]))

# Loss
acc = h['val_loss']
acc_diff = np.diff(acc)
for i in range(len(acc_diff)):
    print("%d, Diff: %.3f, Val: %.6f" % (i, acc_diff[i], acc[i]))


#summarize history for accuracy
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(h['loss'])
plt.plot(h['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()
