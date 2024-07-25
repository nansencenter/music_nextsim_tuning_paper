import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_and_scale(idir, max_date):
    inp_ftrs = pd.read_pickle(f'{idir}/ftrs.pickle')#.drop(columns=['date']).astype(float)
    inp_lbls = pd.read_pickle(f'{idir}/lbls.pickle')#.astype(float)
    inp_rgps = pd.read_pickle(f'{idir}/rgps.pickle')#.drop(columns=['date']).astype(float)
    print(inp_ftrs.shape, inp_lbls.shape, inp_rgps.shape)
    inp_lbls = inp_lbls[inp_ftrs.date < max_date].astype(float)
    inp_ftrs = inp_ftrs[inp_ftrs.date < max_date].drop(columns=['date']).astype(float)
    inp_rgps = inp_rgps[inp_rgps.date < max_date].drop(columns=['date']).astype(float)
    print(inp_ftrs.shape, inp_lbls.shape, inp_rgps.shape)

    ftrs_avg = inp_ftrs.mean()
    ftrs_std = inp_ftrs.std()
    lbls_avg = inp_lbls.mean()
    lbls_std = inp_lbls.std()
    inp_ftrs = (inp_ftrs - ftrs_avg) / ftrs_std
    inp_rgps = (inp_rgps - ftrs_avg) / ftrs_std
    inp_lbls = (inp_lbls - lbls_avg) / lbls_std

    param_names = list(inp_lbls.columns)
    print(param_names)

    return param_names, inp_rgps, inp_ftrs


def get_good_columns1(inp_rgps, max_std):
    rgps_avg = np.median(inp_rgps, axis=0)
    rgps_std = np.std(inp_rgps, axis=0)
    plt.figure(figsize=(15,3))
    arg_idx = np.argsort(rgps_avg)
    rgps_avg = rgps_avg[arg_idx]
    rgps_std = rgps_std[arg_idx]
    columns = inp_rgps.columns[arg_idx]

    gpi = (rgps_avg > -max_std) * (rgps_avg < max_std)

    plt.errorbar(columns, rgps_avg, rgps_std)
    plt.plot(columns[gpi], rgps_avg[gpi], 'r.')
    plt.hlines([-max_std, 0, max_std], 0, len(columns), color='k', alpha=0.5)
    plt.xticks(rotation = 90) # Rotates X-Axis Ticks
    plt.ylabel('Relative scale')
    plt.show()

    good_columns1 = columns[gpi]
    print(len(good_columns1), good_columns1)
    return good_columns1


def build_and_compile_model(input_size, keras):
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(input_size,)),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(input_size)
    ])

    model.compile(
        loss='mean_absolute_error',
        optimizer=keras.optimizers.Adam(0.0005)
    )
    return model

def train_autoenc_predict(inp_ftrs, inp_rgps, good_columns, epochs, keras):
    train_features = inp_ftrs[good_columns].sample(frac=0.75)
    test_features = inp_ftrs[good_columns].drop(train_features.index)

    train_labels = train_features
    test_labels = test_features

    input_size = train_features.shape[1]
    dnn_model = build_and_compile_model(input_size, keras)
    earlystopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    history = dnn_model.fit(
        train_features,
        train_labels,
        validation_data=(test_features, test_labels),
        verbose=0,
        epochs=epochs,
        callbacks=[earlystopping]
    )

    test_predictions = dnn_model.predict(test_features)
    rgps_predictions = dnn_model.predict(inp_rgps[good_columns])
    return test_features, test_predictions, inp_rgps[good_columns], rgps_predictions, history

def plot_rmse(test_features, test_predictions, rgps_features, rgps_predictions, history):
    rmse_n = np.mean((test_predictions - test_features)**2, axis=0)**0.5
    rmse_r = np.mean((rgps_predictions - rgps_features)**2, axis=0)**0.5

    fig, axs = plt.subplots(1,2,figsize=(25,7))
    plot_loss(axs[0], history)

    axs[1].plot(rmse_n, rmse_r, '.')
    for bn, br, c in zip(rmse_n, rmse_r, test_features.columns):
        axs[1].text(bn, br, c)
    plt.show()
    return rmse_n, rmse_r

def plot_loss(ax, history):
    ax.plot(history.history['loss'],  '.-', label='loss')
    ax.plot(history.history['val_loss'], '-', label='val_loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.grid(True)

def plot_all_rmse(rmse_r_list, rmse_n_list, tf_list, tp_list, rf_list, rp_list):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    for rn, rr, tf, tp, rf, rp in zip(rmse_r_list, rmse_n_list, tf_list, tp_list, rf_list, rp_list):
        rer = np.mean(((rf - rp)**2).to_numpy().flatten())
        ter = np.mean(((tf - tp)**2).to_numpy().flatten())
        ax.plot(rn, rr, 'o', alpha=0.5, label=f' {rn.size}, {ter:0.1f}, {rer:0.1f}')

    ax.legend()
    plt.tight_layout()
    plt.show()

def train_params(param_names, inp_ftrs, inp_lbls, inp_rgps, train_func, lbls_std, lbls_avg, n_repeats, epochs, patience, ax=None):
    rgps_pred_params = {}
    test_pred_params = {}
    test_labe_params = {}
    test_prms_params = {}
    train_prms_params = {}

    for param_name in param_names:
        print(param_name)

        rgps_pred_all = []
        test_pred_all = []
        test_labe_all = []
        test_prms_all = []
        train_pred_all = []
        train_prms_all = []

        for i in range(n_repeats):
            train_features = inp_ftrs.sample(frac=0.85)
            test_features = inp_ftrs.drop(train_features.index)

            train_labels = inp_lbls[param_name][train_features.index]
            test_labels = inp_lbls[param_name].drop(train_features.index)
            test_params = inp_lbls.drop(train_features.index)
            train_params = inp_lbls.loc[train_features.index]

            model, history = train_func(train_features, train_labels, test_features, test_labels)

            train_predictions = model.predict(train_features).flatten()
            test_predictions = model.predict(test_features).flatten()
            rgps_predictions = model.predict(inp_rgps).flatten()

            train_predictions = train_predictions * lbls_std[param_name] + lbls_avg[param_name]
            test_predictions = test_predictions * lbls_std[param_name] + lbls_avg[param_name]
            train_labels = train_labels * lbls_std[param_name] + lbls_avg[param_name]
            test_labels = test_labels * lbls_std[param_name] + lbls_avg[param_name]
            test_params = test_params * lbls_std + lbls_avg
            train_params = train_params * lbls_std + lbls_avg
            rgps_predictions = rgps_predictions * lbls_std[param_name] + lbls_avg[param_name]

            test_pred_all.append(test_predictions)
            rgps_pred_all.append(rgps_predictions)
            test_labe_all.append(test_labels)
            test_prms_all.append(test_params)
            train_prms_all.append(train_params)
            train_pred_all.append(train_predictions)

            if ax and history:
                plot_loss(ax, history)

        rgps_pred_params[param_name] = rgps_pred_all
        test_pred_params[param_name] = test_pred_all
        test_labe_params[param_name] = test_labe_all
        test_prms_params[param_name] = test_prms_all
        train_prms_params[param_name] = train_prms_all

    return rgps_pred_params, test_pred_params, test_labe_params, test_prms_params, train_prms_params

def plot_scatter_histo(param_names, test_labe_params, test_pred_params, rgps_pred_params, bins, density, xlims):
    for param_name in param_names:
        fig, axs = plt.subplots(1, 2, figsize=(15,5))
        for test_labels, test_predictions in zip(test_labe_params[param_name], test_pred_params[param_name]):
            axs[0].plot(test_labels, test_predictions, 'b.', alpha=0.1)
            axs[0].set_ylabel('Predictions')
            axs[0].set_xlabel(param_name)
            axs[0].set_aspect('equal')
            axs[0].plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], '-k')
        if param_name in xlims:
            therange = xlims[param_name]
        else:
            therange = None
        axs[1].hist(np.hstack(test_pred_params[param_name]), bins, range=therange, density=density, alpha=0.3, label='test')
        axs[1].hist(np.hstack(rgps_pred_params[param_name]), bins, range=therange, density=density, alpha=0.3, label='test')
        plt.show()