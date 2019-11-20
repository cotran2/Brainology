from models import ctc_model
from utils import *
from sklearn.model_selection import train_test_split
from jiwer import wer as jwer


class parameters():
    freq = 100
    feature = 1
    number_sentence = 3
    batch_size = 100
    tcn = True
    data = 'eeg'
    padding_value = float(255)
    batch_size = 100
    epochs = 200
    nb_labels = ord('z') - ord('a') + 1 + 1 + 1

if __name__ == '__main__':


    params = parameters
    input_set, target_set, seq_len_set, original_set = load_data(params)
    input_set, input_length = pad_sequences(input_set,dtype='float32',
                                            value=params.padding_value)
    target_set, target_length = pad_sequences(target_set,dtype='int32',
                                              value=params.padding_value)
    print("Input shape {} ; Target shape {}".format(input_set.shape,target_set.shape))

    x_train,x_test,y_train,y_test = train_test_split(input_set,target_set,test_size=0.1,
                                                     random_state=45)
    xl_train,xl_test,yl_train,yl_test = train_test_split(input_length,target_length,test_size=0.1,
                                                     random_state=45)
    ctc_model = ctc_model(input_set.shape[-1],params.nb_labels,params.padding_value,False)
    ctc_model.fit(x=[x_train, y_train, xl_train, yl_train],
                  y=np.zeros(len(x_train)),
                batch_size=params.batch_size, epochs=params.epochs,
                validation_split=0.1)
    pred = ctc_model.predict([x_test, xl_test],
                           batch_size = params.batch_size,
                           max_value=int(params.padding_value))

    y_test_depad = unpadding(y_test,yl_test)
    pred = inverse_ctc_format(pred)
    y_test_depad = inverse_ctc_format(y_test_depad)
    print(jwer(pred,y_test_depad))