train.shape[0], seq_length, img_ht * img_width * 3)
# features_test = features_test.reshape(features_test.shape[0], seq_length, img_ht * img_width * 3)

# def reshape_data(data):
#     return np.reshape(data, (data.shape[0], seq_length, -1))

# features_train = reshape_data(features_train)
# features_test = reshape_data(features_test)