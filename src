import os
import sys

def predict(begin_date, end_date, model_name, src='radar'):
    config = get_config('test', src=src)
    save_dir = config['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_name = model_name
    print('Load model:' + model_path + '/' + model_name)
    model = FactorWeatherModel.load(os.path.join(model_path, model_name))  # cPickle.load()
    print("done")
    model.set_mode("predict")
    predict_func = theano.function(inputs=model.interface_layer.input_symbols(),
                          outputs=sparnn.utils.quick_reshape_patch_back(model.middle_layers[-1].output,
                                                           config['patch_size']),
                          on_unused_input='ignore')
    it = begin_date  # real time the predict is one moment
    while(it <=end_date):
        it += datetime.timedelta(minutes=config['interval']) # predict move time long?  this predict distance next predict how long time
        start_date = it - datetime.timedelta(minutes=(config['input_seq_length']-1)*config['interval'])
        config['start_date'] = start_date
        config['end_date'] = it
        print('loading data', config['start_date'], config['end_date'])
        try:
            X = read_X(config)
            test_iterator = load_data(X,config, mode='predict')
            test_iterator.begin(do_shuffle=False)
            print(0)
        except Exception as e:
            print(Exception, e)
            continue
        #result = predict_func(*(test_iterator.input_batch()))*config['max']-config['offset']
        result = predict_func(*(test_iterator.input_batch()))*(config['max']-config['dmin'])+config['dmin']
	    print(result.shape)
        result_dir = os.path.join(save_dir, it.strftime('%Y%m%d%H%M')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        input_image = np.reshape(test_iterator.input_batch()[0][-1][0],
                                 (1, config['size'][0], config['size'][1]))[0]*(config['max']-config['dmin'])+config['dmin']
        write_image(input_image, result_dir, it, config)  # picture
        print('predict', it, result.shape, input_image.max(), input_image.min(), result.max(), result.min())
        for i, r in enumerate(result):
            image = np.reshape(r[0], (1, config['size'][0], config['size'][1]))[0]
            write_image(image, result_dir, it, config, predict=i)  # picture
