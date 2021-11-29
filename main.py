import numpy as np
from help_functions import *
from models import *
from superposition import *
from prepare_data import *
from torchinfo import summary
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    superposition = True
    first_average = 'average'     # show results on 'first' task or the 'average' results until current task

    use_MLP = True      # if True use MLP, else use Transformer
    input_size = 32
    num_heads = 4
    num_layers = 1
    dim_feedforward = 1024
    num_classes = 2
    standardize_input = False

    batch_size = 128
    num_runs = 3
    num_tasks = 3
    num_epochs = 50 if use_MLP else 10
    learning_rate = 0.001

    # # save X, y, mask for all three datasets
    # X, y, mask = preprocess_hate_speech('datasets/hate_speech.csv')
    # torch.save(X, 'Word2Vec_embeddings/X_hate_speech.pt')
    # torch.save(y, 'Word2Vec_embeddings/y_hate_speech.pt')
    # torch.save(mask, 'Word2Vec_embeddings/mask_hate_speech.pt')
    #
    # X, y, mask = preprocess_IMDB_reviews('datasets/IMDB_sentiment_analysis.csv')
    # torch.save(X, 'Word2Vec_embeddings/X_IMDB_sentiment_analysis.pt')
    # torch.save(y, 'Word2Vec_embeddings/y_IMDB_sentiment_analysis.pt')
    # torch.save(mask, 'Word2Vec_embeddings/mask_IMDB_sentiment_analysis.pt')
    #
    # X, y, mask = preprocess_SMS_spam('datasets/sms_spam.csv')
    # torch.save(X, 'Word2Vec_embeddings/X_sms_spam.pt')
    # torch.save(y, 'Word2Vec_embeddings/y_sms_spam.pt')
    # torch.save(mask, 'Word2Vec_embeddings/mask_sms_spam.pt')

    # model = Transformer(input_size, num_heads, num_layers, dim_feedforward, num_classes).cuda()
    # x = model(X[:64].cuda(), mask[:64].cuda())
    #
    # print(model)
    # summary(model, [(batch_size, 256, 32), (batch_size, 256)])
    # print('Number of trainable parameters: ', count_trainable_parameters(model))

    # Train model for 'num_runs' runs for 'num_tasks' tasks
    acc_arr = np.zeros((num_runs, num_tasks))
    auroc_arr = np.zeros((num_runs, num_tasks))
    auprc_arr = np.zeros((num_runs, num_tasks))

    acc_epoch = np.zeros((num_runs, num_tasks * num_epochs))
    auroc_epoch = np.zeros((num_runs, num_tasks * num_epochs))
    auprc_epoch = np.zeros((num_runs, num_tasks * num_epochs))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for r in range(num_runs):
        print('- - Run %d - -' % (r + 1))

        if use_MLP:
            model = MLP(input_size, num_classes).to(device)
        else:
            model = MyTransformer(input_size, num_heads, num_layers, dim_feedforward, num_classes).to(device)

        all_tasks_test_data = []
        contexts, layer_dimension = create_context_vectors(model, num_tasks)

        for t in range(num_tasks):
            print('- Task %d -' % (t + 1))

            criterion = torch.nn.CrossEntropyLoss().cuda()

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2,
                                                                   threshold=0.0001, min_lr=1e-8, verbose=True)

            print('Number of trainable parameters: ', count_trainable_parameters(model))
            # print(model)
            # summary(model, [(batch_size, 256, 32), (batch_size, 256)])

            best_auroc_val = 0

            # prepare data
            if t == 0:
                X = torch.load('Word2Vec_embeddings/X_hate_speech.pt').float()
                y = torch.load('Word2Vec_embeddings/y_hate_speech.pt')
                mask = torch.load('Word2Vec_embeddings/mask_hate_speech.pt').float()
            elif t == 1:
                X = torch.load('Word2Vec_embeddings/X_IMDB_sentiment_analysis.pt').float()
                y = torch.load('Word2Vec_embeddings/y_IMDB_sentiment_analysis.pt')
                mask = torch.load('Word2Vec_embeddings/mask_IMDB_sentiment_analysis.pt').float()
            elif t == 2:
                X = torch.load('Word2Vec_embeddings/X_sms_spam.pt').float()
                y = torch.load('Word2Vec_embeddings/y_sms_spam.pt')
                mask = torch.load('Word2Vec_embeddings/mask_sms_spam.pt').float()

            if standardize_input:
                for i in range(X.shape[0]):
                    X[i, :, :] = torch.from_numpy(StandardScaler().fit_transform(X[i, :, :]))

                    # where samples are padded, make zeros again
                    mask_i = torch.ones(X.shape[1]) - mask[i, :]
                    for j in range(X.shape[2]):
                        X[i, :, j] = X[i, :, j] * mask_i

            # split data into train, validation and test set
            y = torch.max(y, 1)[1]  # change one-hot-encoded vectors to numbers
            permutation = torch.randperm(X.size()[0])
            X = X[permutation]
            y = y[permutation]
            mask = mask[permutation]
            index_val = round(0.8 * len(permutation))
            index_test = round(0.9 * len(permutation))

            X_train, y_train, mask_train = X[:index_val, :, :].to(device), y[:index_val].to(device), mask[:index_val, :].to(device)   # data to device
            X_val, y_val, mask_val = X[index_val:index_test, :, :], y[index_val:index_test], mask[index_val:index_test, :]
            X_test, y_test, mask_test = X[index_test:, :, :], y[index_test:], mask[index_test:, :]

            all_tasks_test_data.append([X_test, y_test, mask_test])

            for epoch in range(num_epochs):
                model.train()
                model = model.cuda()

                permutation = torch.randperm(X_train.size()[0])

                train_outputs = []
                permuted_y = []
                for i in range(0, X_train.size()[0], batch_size):
                    indices = permutation[i:i + batch_size]
                    batch_X, batch_y, batch_mask = X_train[indices], y_train[indices], mask_train[indices]
                    permuted_y.append(batch_y)

                    if use_MLP:
                        outputs = model.forward(batch_X)
                    else:
                        outputs = model.forward(batch_X, batch_mask)
                    train_outputs.append(outputs)

                    optimizer.zero_grad()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                train_acc, train_auroc, train_auprc = get_stats(train_outputs, torch.cat(permuted_y, dim=0))

                print("Epoch: %d --- train acc: %.2f, train AUROC: %.2f, train AUPRC: %.2f" %
                      (epoch, train_acc * 100, train_auroc * 100, train_auprc * 100))

                # check validation set
                model.eval()
                with torch.no_grad():
                    model = model.cpu()     # model to CPU because of GPU memory restrictions

                    val_outputs = []
                    for i in range(0, X_val.size()[0], batch_size):
                        batch_X, batch_mask = X_val[i:i + batch_size], mask_val[i:i + batch_size]

                        if use_MLP:
                            outputs = model.forward(batch_X)
                        else:
                            outputs = model.forward(batch_X, batch_mask)

                        val_outputs.append(outputs)

                    val_acc, val_auroc, val_auprc = get_stats(val_outputs, y_val)
                    val_loss = criterion(torch.cat(val_outputs, dim=0), y_val)

                    print("Epoch: %d --- val acc: %.2f, val AUROC: %.2f, val AUPRC: %.2f, val loss: %.3f" %
                          (epoch, val_acc * 100, val_auroc * 100, val_auprc * 100, val_loss))

                    scheduler.step(val_auroc)
                    if val_auroc > best_auroc_val:
                        best_auroc_val = val_auroc
                        torch.save(model.state_dict(), 'models/model_best.pt')

                # todo
                # track results with or without superposition
                acc_e, auroc_e, auprc_e = evaluate_results(model, contexts, layer_dimension, all_tasks_test_data,
                                                           superposition, t, first_average, use_MLP, batch_size)

                acc_epoch[r, (t * num_epochs) + epoch] = acc_e
                auroc_epoch[r, (t * num_epochs) + epoch] = auroc_e
                auprc_epoch[r, (t * num_epochs) + epoch] = auprc_e

            # check test set
            model.load_state_dict(torch.load('models/model_best.pt'))
            model.eval()
            with torch.no_grad():
                test_outputs = []
                for i in range(0, X_test.size()[0], batch_size):
                    batch_X, batch_mask = X_test[i:i + batch_size], mask_test[i:i + batch_size]

                    if use_MLP:
                        outputs = model.forward(batch_X)
                    else:
                        outputs = model.forward(batch_X, batch_mask)

                    test_outputs.append(outputs)

                test_acc, test_auroc, test_auprc = get_stats(test_outputs, y_test)

                print("TEST: test acc: %.2f, test AUROC: %.2f, test AUPRC: %.2f" %
                      (test_acc * 100, test_auroc * 100, test_auprc * 100))

                predicted = np.argmax(torch.cat(test_outputs, dim=0).cpu().detach().numpy(), axis=1).ravel()
                # print('Classification report:', classification_report(y_test.cpu().detach().numpy(), predicted))
                print('Confusion matrix:\n', confusion_matrix(y_test.cpu().detach().numpy(), predicted, labels=list(range(num_classes))))

            # store statistics
            acc_arr[r, t] = test_acc * 100
            auroc_arr[r, t] = test_auroc * 100
            auprc_arr[r, t] = test_auprc * 100

            if superposition:   # perform context multiplication
                if t < num_tasks - 1:   # do not multiply with contexts at the end of last task
                    context_multiplication(model, contexts, layer_dimension, t)

    # display mean and standard deviation per task
    mean_acc, std_acc = np.mean(acc_arr, axis=0), np.std(acc_arr, axis=0)
    mean_auroc, std_auroc = np.mean(auroc_arr, axis=0), np.std(auroc_arr, axis=0)
    mean_auprc, std_auprc = np.mean(auprc_arr, axis=0), np.std(auprc_arr, axis=0)

    for t in range(num_tasks):
        if t == 0:
            s = 'Hate speech'
        elif t == 1:
            s = 'IMDB sentiment analysis'
        elif t == 2:
            s = 'SMS spam'

        print('------------------------------------------')
        print('%s - Accuracy = %.1f +/- %.1f' % (s, mean_acc[t], std_acc[t]))
        print('%s - AUROC    = %.1f +/- %.1f' % (s, mean_auroc[t], std_auroc[t]))
        print('%s - AUPRC    = %.1f +/- %.1f' % (s, mean_auprc[t], std_auprc[t]))

    # display mean and standard deviation per epoch
    mean_acc, std_acc = np.mean(acc_epoch, axis=0), np.std(acc_epoch, axis=0)
    mean_auroc, std_auroc = np.mean(auroc_epoch, axis=0), np.std(auroc_epoch, axis=0)
    mean_auprc, std_auprc = np.mean(auprc_epoch, axis=0), np.std(auprc_epoch, axis=0)

    show_only_accuracy = False
    if show_only_accuracy:
        plot_multiple_results([mean_acc], [std_acc], ['Accuracy'],
                              '%s task results, %s model, %s' % (first_average, 'MLP' if use_MLP else 'Transformer',
                                                                 'superposition' if superposition else 'no superposition'),
                              ['tab:blue'], 'Epoch', 'Accuracy (%)',
                              [(i + 1) * num_epochs for i in range(num_tasks - 1)], 0, 100)
    else:   # show all three metrics
        plot_multiple_results([mean_acc, mean_auroc, mean_auprc], [std_acc, std_auroc, std_auprc], ['Accuracy', 'AUROC', 'AUPRC'],
                              '%s task results, %s model, %s' % (first_average, 'MLP' if use_MLP else 'Transformer', 'superposition' if superposition else 'no superposition'),
                              ['tab:blue', 'tab:orange', 'tab:green'], 'Epoch', 'Metric value',
                              [(i + 1) * num_epochs for i in range(num_tasks - 1)], 0, 100)



