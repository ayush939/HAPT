import gin
import sklearn.metrics
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
import os
import sklearn
import seaborn
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from evaluation.metrics import ConfusionMatrix


@gin.configurable
def evaluate(model, checkpoint, ds_test, checkpoint_path, run_paths, num_classes):

    confusion_matrix_test = ConfusionMatrix(num_classes=num_classes)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    def _test_step(images, labels):

        """
        evaluate a single step for given images and labels
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        
        test_loss(t_loss)
        test_accuracy(labels, predictions)
        #visualize_predictions(predictions, images, labels)
        predictions = tf.math.argmax(predictions, -1)
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
        confusion_matrix_test.update_state(labels, predictions)

        return predictions, labels

    
    for idx, (images, labels) in enumerate(ds_test):
            step = idx + 1
            predictions, labels = _test_step(images, labels)
            visualize_predictions(predictions, images, labels)
            
            
    template = 'Test Loss: {}, Test Accuracy: {}'
    logging.info(template.format(test_loss.result(),test_accuracy.result() * 100))

    
    template = 'Confusion Matrix: \n{}'
    logging.info(template.format(confusion_matrix_test.result().numpy()))

    
    # Confusion Matrix Visualization
    plt.figure(figsize=(24, 24))
    cm = np.array(confusion_matrix_test.result().numpy().tolist())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    seaborn.set(font_scale=2)
    seaborn.heatmap(cm, annot=True, fmt='.1%')
    plt.xlabel('Predict')
    plt.ylabel('True')
    #cm_path = os.path.join(run_paths['path_summary_image'], 'confusion_matrix.png')
    plt.savefig('./confusion1.png')

    
def visualize_predictions(total_predictions, total_signal,total_label):

    '''
    This function visualizes the labels of the test sequence predicted. 
    '''

    total_label = tf.reshape(total_label, [-1]).numpy()
    total_predictions = tf.math.argmax(total_predictions, -1)
    total_predictions = tf.reshape(total_predictions, [-1]).numpy()
    total_signal = tf.reshape(total_signal, [-1, 6]).numpy()

    x=range(8000)

    ax1 = plt.subplot(4, 1, 1)
    plt.title('Predicted label')


    index = 0
    for idx, predictions in np.ndenumerate(total_predictions):
        index += 1
        if index <= 8000:
            if predictions == 0:
                plt.axvline(x=index, color='#FF3700', alpha=0.1)
            if predictions == 1:
                plt.axvline(x=index, color='#FF6E00', alpha=0.1)
            if predictions == 2:
                plt.axvline(x=index, color='#FFA500', alpha=0.1)
            if predictions == 3:
                plt.axvline(x=index, color='#BEAA26', alpha=0.1)
            if predictions == 4:
                plt.axvline(x=index, color='#7DAE4B', alpha=0.1)
            if predictions == 5:
                plt.axvline(x=index, color='#3CB371', alpha=0.1)
            if predictions == 6:
                plt.axvline(x=index, color='#28AF8F', alpha=0.1)
            if predictions == 7:
                plt.axvline(x=index, color='#14AAAD', alpha=0.1)
            if predictions == 8:
                plt.axvline(x=index, color='#00A6CB', alpha=0.1)
            if predictions == 9:
                plt.axvline(x=index, color='#238DCC', alpha=0.1)
            if predictions == 10:
                plt.axvline(x=index, color='#4773CC', alpha=0.1)
            if predictions == 11:
                plt.axvline(x=index, color='#1f77b4', alpha=0.1)
            if predictions == 12:
                plt.axvline(x=index, color='#6A5ACD', alpha=0.1)
    
    #plt.subplot(4, 1, 4)
    plt.bar(1, 1, alpha=0.5, width=1, facecolor='#FF3700', edgecolor='white',  lw=1)
    plt.bar(2, 1, alpha=0.5, width=1, facecolor='#FF6E00', edgecolor='white', lw=1)
    plt.bar(3, 1, alpha=0.5, width=1, facecolor='#FFA500', edgecolor='white', lw=1)
    plt.bar(4, 1, alpha=0.5, width=1, facecolor='#BEAA26', edgecolor='white', lw=1)
    plt.bar(5, 1, alpha=0.5, width=1, facecolor='#7DAE4B', edgecolor='white', lw=1)
    plt.bar(6, 1, alpha=0.5, width=1, facecolor='#3CB371', edgecolor='white', lw=1)
    plt.bar(7, 1, alpha=0.5, width=1, facecolor='#28AF8F', edgecolor='white', lw=1)
    plt.bar(8, 1, alpha=0.5, width=1, facecolor='#14AAAD', edgecolor='white', lw=1)
    plt.bar(9, 1, alpha=0.5, width=1, facecolor='#00A6CB', edgecolor='white', lw=1)
    plt.bar(10, 1, alpha=0.5, width=1, facecolor='#238DCC', edgecolor='white', lw=1)
    plt.bar(11, 1, alpha=0.5, width=1, facecolor='#4773CC', edgecolor='white', lw=1)
    plt.bar(12, 1, alpha=0.5, width=1, facecolor='#1f77b4', edgecolor='white', lw=1)

    plt.xticks([1, 2, 3, 4, 5,6,7,8,9,10,11,12], ['WALKING 0', 'WALKING_UPSTAIRS 1', 'WALKING_DOWNSTAIRS 2', 'SITTING 3', 'STANDING 4', 'LAYING 5', 'STAND_TO_SIT 6', 'SIT_TO_STAND 7', 'SIT_TO_LIE 8', 'LIT_TO_SIT 9','STAND_TO_LIE 10','LIE_TO_STAND 11'])
    plt.xticks(rotation=90)
    
    plt.savefig('./labels.png')				