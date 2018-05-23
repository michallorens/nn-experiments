def metric_learn(metric, model, data_set):
    model.eval()
    for batch, inputs in enumerate(data_set):
        labels = inputs[0][0]
        inputs = [i[1].cuda() for i in inputs]
        inputs, _, _ = model(*inputs)
        features = inputs.numpy()
        labels = labels.numpy()
        metric.fit(features, labels)
