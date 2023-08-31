# Human-Posture-Classification
An integrated network combines GCN+LSTM+TransfermorEncoderLayer for Human Posture Classification, which achieves an averageaccuracy of 80%.

Dataset Source: [https://github.com/hamlinzheng/Awesome-Skeleton-based-Action-Recognition] -- (2017) SYSU 3D Human-Object Interaction Dataset (SYSU).
Data Description: Each sample is a Numpy array of size (1,3,128,17,2), representing a batchsize of 1, 3-dimensional coordinates, 128 frames, 17 joints, and 2 people. However, in this dataset, only the 2-dimensional coordinates are valid, i.e., “sample[:, 1, :, :, :] and sample[:, :, :, :, :, 1]” can be assigned a value of 0 or deleted (I deleted them). I did not make any normalize process.

Note: Remember to upzip "data.7z"!

Challenges:
C1. CNN/RNN/GCN/Transformer are applicable to this task, how to achieve the best effect? 

Adopting the integration of RNN, GCN and Transformer to extract spatio-temporal features.

C2. How to deal with a small sample size: Is generation or data enhancement worth trying?

DataFrame extraction. The dataframes of the original samples will be drawn every step size and integrated into a new sample with shape [1, 2, 48, 17].
