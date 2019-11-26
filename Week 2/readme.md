## Week 2 Assignment:

### Log of final model:

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 7s 123us/step - loss: 0.3170 - acc: 0.8858 - val_loss: 0.0761 - val_acc: 0.9782
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 6s 92us/step - loss: 0.1408 - acc: 0.9420 - val_loss: 0.0444 - val_acc: 0.9871
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 6s 92us/step - loss: 0.1196 - acc: 0.9480 - val_loss: 0.0326 - val_acc: 0.9901
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 5s 90us/step - loss: 0.1087 - acc: 0.9505 - val_loss: 0.0387 - val_acc: 0.9893
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 5s 91us/step - loss: 0.1075 - acc: 0.9492 - val_loss: 0.0248 - val_acc: 0.9918
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0991 - acc: 0.9541 - val_loss: 0.0271 - val_acc: 0.9908
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 5s 90us/step - loss: 0.0940 - acc: 0.9543 - val_loss: 0.0326 - val_acc: 0.9904
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0932 - acc: 0.9543 - val_loss: 0.0254 - val_acc: 0.9935
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0907 - acc: 0.9559 - val_loss: 0.0248 - val_acc: 0.9936
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0890 - acc: 0.9549 - val_loss: 0.0254 - val_acc: 0.9931
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0851 - acc: 0.9558 - val_loss: 0.0261 - val_acc: 0.9935
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0865 - acc: 0.9564 - val_loss: 0.0255 - val_acc: 0.9933
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0823 - acc: 0.9581 - val_loss: 0.0226 - val_acc: 0.9942
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0823 - acc: 0.9572 - val_loss: 0.0239 - val_acc: 0.9932
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 92us/step - loss: 0.0821 - acc: 0.9578 - val_loss: 0.0230 - val_acc: 0.9942
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 6s 93us/step - loss: 0.0821 - acc: 0.9579 - val_loss: 0.0239 - val_acc: 0.9936
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0813 - acc: 0.9576 - val_loss: 0.0217 - val_acc: 0.9941
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0791 - acc: 0.9592 - val_loss: 0.0262 - val_acc: 0.9932
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 5s 90us/step - loss: 0.0790 - acc: 0.9588 - val_loss: 0.0254 - val_acc: 0.9938
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 5s 91us/step - loss: 0.0792 - acc: 0.9588 - val_loss: 0.0233 - val_acc: 0.9944
Out[14]:
<keras.callbacks.History at 0x7f05d8e5f128>


### Result of model.evaluate

*score = model3.evaluate(X_test, Y_test, verbose=0)
print(score)*
[0.023280380787370086, 0.9944]


### Strategy

**First step** was to remove the use of bias in each Conv2D layer.
The **second step** was to see if there was any leeway for reduction of kernels in a few layers - reduced the number of layers in the 4th and 5th conv2D layers (from 16 each to 12 and 14 respectively) without any significant hit to the validation accuracy.
**Third step -** Since, the max validation accuracy after the second step was less than 99.4%, there was a need to increase the variance of the model. This was done by relaxing the regularization in the model, by removing batch-normalization in final 2 layers.
