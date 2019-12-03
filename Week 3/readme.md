**Final Validation Accuracy for base network:**
 82.49%  
  
**Model Definition:**  
```
# Define the model
model7 = Sequential()

# 3x3 depthwise separable conv2d with same padding========================================================================
model7.add(SeparableConv2D(filters = 128, kernel_size=3, padding= 'same', activation = 'relu', input_shape = (32, 32, 3)))
# output of above layer - 32x32x128, receptive field: 3x3
model7.add(BatchNormalization())
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with dropout=============================================================================
model7.add(SeparableConv2D(filters = 128, kernel_size=3, activation = 'relu'))
# output of above layer - 30x30x128, receptive field: 5x5
model7.add(BatchNormalization())
model7.add(Dropout(0.2))
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with dropout=============================================================================
model7.add(SeparableConv2D(filters = 128, kernel_size=3, activation = 'relu'))
# output of above layer - 28x28x128, receptive field: 7x7
model7.add(BatchNormalization())
model7.add(Dropout(0.2))
# ------------------------------------------------------------------------------------------------------------------------

# max-pool================================================================================================================
model7.add(MaxPooling2D(pool_size=(2, 2)))
# output of above layer - 14x14x128, receptive field: 14x14
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with same padding and dropout============================================================
model7.add(SeparableConv2D(filters = 80, kernel_size=3, padding = 'same', activation = 'relu'))
# output of above layer - 14x14x80, receptive field: 16x16
model7.add(BatchNormalization())
model7.add(Dropout(0.2))
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with dropout=============================================================================
model7.add(SeparableConv2D(filters = 96, kernel_size=3, activation = 'relu'))
# output of above layer - 12x12x96, receptive field: 18x18
model7.add(BatchNormalization())
model7.add(Dropout(0.2))
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with dropout=============================================================================
model7.add(SeparableConv2D(filters = 128, kernel_size=3, activation = 'relu'))
# output of above layer - 10x10x128, receptive field: 20x20
model7.add(BatchNormalization())
model7.add(Dropout(0.3))
# ------------------------------------------------------------------------------------------------------------------------

# max-pool================================================================================================================
model7.add(MaxPooling2D(pool_size=(2, 2)))
# output of above layer - 5x5x128, receptive field: 32x32 (40x40 theoretically)
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 depthwise separable conv2d with dropout=============================================================================
model7.add(SeparableConv2D(filters = 160, kernel_size=3, activation = 'relu'))
# output of above layer - 3x3x160, receptive field: 32x32 (42x42 theoretically)
model7.add(Dropout(0.3))
# ------------------------------------------------------------------------------------------------------------------------

# 3x3 conv2D with softmax activation======================================================================================
model7.add(SeparableConv2D(filters = num_classes, kernel_size=3, activation = 'softmax'))
# output of above layer - 1x1xnum_classes(10), receptive field: 32x32 (44x44 theoretically)
# ------------------------------------------------------------------------------------------------------------------------

# flatten=================================================================================================================
model7.add(Flatten())
# ------------------------------------------------------------------------------------------------------------------------

# Compile the model
model7.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```


**Log:**  

```
Epoch 1/50
390/390 [==============================] - 47s 120ms/step - loss: 1.4698 - acc: 0.4576 - val_loss: 1.1835 - val_acc: 0.5827
Epoch 2/50
390/390 [==============================] - 41s 104ms/step - loss: 0.9883 - acc: 0.6485 - val_loss: 0.9754 - val_acc: 0.6625
Epoch 3/50
390/390 [==============================] - 41s 104ms/step - loss: 0.8345 - acc: 0.7051 - val_loss: 0.8313 - val_acc: 0.7094
Epoch 4/50
390/390 [==============================] - 41s 105ms/step - loss: 0.7404 - acc: 0.7399 - val_loss: 0.7503 - val_acc: 0.7431
Epoch 5/50
390/390 [==============================] - 41s 104ms/step - loss: 0.6759 - acc: 0.7630 - val_loss: 0.7719 - val_acc: 0.7356
Epoch 6/50
390/390 [==============================] - 40s 102ms/step - loss: 0.6291 - acc: 0.7765 - val_loss: 0.6960 - val_acc: 0.7608
Epoch 7/50
390/390 [==============================] - 41s 104ms/step - loss: 0.5975 - acc: 0.7887 - val_loss: 0.6714 - val_acc: 0.7735
Epoch 8/50
390/390 [==============================] - 40s 103ms/step - loss: 0.5634 - acc: 0.8004 - val_loss: 0.6411 - val_acc: 0.7793
Epoch 9/50
390/390 [==============================] - 41s 105ms/step - loss: 0.5382 - acc: 0.8106 - val_loss: 0.6176 - val_acc: 0.7890
Epoch 10/50
390/390 [==============================] - 41s 104ms/step - loss: 0.5134 - acc: 0.8201 - val_loss: 0.6566 - val_acc: 0.7786
Epoch 11/50
390/390 [==============================] - 41s 106ms/step - loss: 0.4974 - acc: 0.8248 - val_loss: 0.6372 - val_acc: 0.7824
Epoch 12/50
390/390 [==============================] - 40s 104ms/step - loss: 0.4784 - acc: 0.8311 - val_loss: 0.6647 - val_acc: 0.7772
Epoch 13/50
390/390 [==============================] - 41s 104ms/step - loss: 0.4580 - acc: 0.8389 - val_loss: 0.5590 - val_acc: 0.8141
Epoch 14/50
390/390 [==============================] - 40s 103ms/step - loss: 0.4443 - acc: 0.8439 - val_loss: 0.5971 - val_acc: 0.8023
Epoch 15/50
390/390 [==============================] - 41s 104ms/step - loss: 0.4343 - acc: 0.8471 - val_loss: 0.6494 - val_acc: 0.7910
Epoch 16/50
390/390 [==============================] - 40s 103ms/step - loss: 0.4220 - acc: 0.8511 - val_loss: 0.5595 - val_acc: 0.8169
Epoch 17/50
390/390 [==============================] - 41s 104ms/step - loss: 0.4093 - acc: 0.8560 - val_loss: 0.5795 - val_acc: 0.8099
Epoch 18/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3969 - acc: 0.8582 - val_loss: 0.5966 - val_acc: 0.8036
Epoch 19/50
390/390 [==============================] - 41s 106ms/step - loss: 0.3964 - acc: 0.8612 - val_loss: 0.5990 - val_acc: 0.8052
Epoch 20/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3814 - acc: 0.8657 - val_loss: 0.5647 - val_acc: 0.8170
Epoch 21/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3702 - acc: 0.8670 - val_loss: 0.5772 - val_acc: 0.8144
Epoch 22/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3707 - acc: 0.8689 - val_loss: 0.5692 - val_acc: 0.8116
Epoch 23/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3547 - acc: 0.8754 - val_loss: 0.6537 - val_acc: 0.8037
Epoch 24/50
390/390 [==============================] - 40s 104ms/step - loss: 0.3504 - acc: 0.8761 - val_loss: 0.5654 - val_acc: 0.8220
Epoch 25/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3461 - acc: 0.8770 - val_loss: 0.5346 - val_acc: 0.8312
Epoch 26/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3410 - acc: 0.8783 - val_loss: 0.5945 - val_acc: 0.8166
Epoch 27/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3344 - acc: 0.8796 - val_loss: 0.5842 - val_acc: 0.8213
Epoch 28/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3257 - acc: 0.8844 - val_loss: 0.5596 - val_acc: 0.8216
Epoch 29/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3201 - acc: 0.8854 - val_loss: 0.5881 - val_acc: 0.8160
Epoch 30/50
390/390 [==============================] - 40s 103ms/step - loss: 0.3098 - acc: 0.8878 - val_loss: 0.6156 - val_acc: 0.8121
Epoch 31/50
390/390 [==============================] - 41s 105ms/step - loss: 0.3086 - acc: 0.8903 - val_loss: 0.6281 - val_acc: 0.8182
Epoch 32/50
390/390 [==============================] - 41s 104ms/step - loss: 0.3058 - acc: 0.8907 - val_loss: 0.6335 - val_acc: 0.8172
Epoch 33/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2996 - acc: 0.8925 - val_loss: 0.5881 - val_acc: 0.8232
Epoch 34/50
390/390 [==============================] - 41s 105ms/step - loss: 0.2991 - acc: 0.8935 - val_loss: 0.5898 - val_acc: 0.8269
Epoch 35/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2950 - acc: 0.8948 - val_loss: 0.5936 - val_acc: 0.8192
Epoch 36/50
390/390 [==============================] - 40s 104ms/step - loss: 0.2877 - acc: 0.8964 - val_loss: 0.5584 - val_acc: 0.8365
Epoch 37/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2861 - acc: 0.8975 - val_loss: 0.5900 - val_acc: 0.8306
Epoch 38/50
390/390 [==============================] - 40s 103ms/step - loss: 0.2774 - acc: 0.9004 - val_loss: 0.5932 - val_acc: 0.8250
Epoch 39/50
390/390 [==============================] - 41s 105ms/step - loss: 0.2774 - acc: 0.9008 - val_loss: 0.5709 - val_acc: 0.8322
Epoch 40/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2730 - acc: 0.9019 - val_loss: 0.6186 - val_acc: 0.8233
Epoch 41/50
390/390 [==============================] - 41s 105ms/step - loss: 0.2692 - acc: 0.9039 - val_loss: 0.6259 - val_acc: 0.8201
Epoch 42/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2742 - acc: 0.9018 - val_loss: 0.6726 - val_acc: 0.8093
Epoch 43/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2561 - acc: 0.9068 - val_loss: 0.5955 - val_acc: 0.8314
Epoch 44/50
390/390 [==============================] - 40s 103ms/step - loss: 0.2593 - acc: 0.9063 - val_loss: 0.6114 - val_acc: 0.8253
Epoch 45/50
390/390 [==============================] - 40s 103ms/step - loss: 0.2593 - acc: 0.9069 - val_loss: 0.6199 - val_acc: 0.8249
Epoch 46/50
390/390 [==============================] - 40s 103ms/step - loss: 0.2593 - acc: 0.9061 - val_loss: 0.6060 - val_acc: 0.8323
Epoch 47/50
390/390 [==============================] - 40s 103ms/step - loss: 0.2518 - acc: 0.9088 - val_loss: 0.6087 - val_acc: 0.8308
Epoch 48/50
390/390 [==============================] - 41s 105ms/step - loss: 0.2471 - acc: 0.9124 - val_loss: 0.6385 - val_acc: 0.8252
Epoch 49/50
390/390 [==============================] - 40s 104ms/step - loss: 0.2477 - acc: 0.9102 - val_loss: 0.5918 - val_acc: 0.8379
Epoch 50/50
390/390 [==============================] - 41s 104ms/step - loss: 0.2426 - acc: 0.9114 - val_loss: 0.6393 - val_acc: 0.8224
```
