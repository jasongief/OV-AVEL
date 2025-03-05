import numpy as np
import pdb


def Precision(X_pre, X_gt):

    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x*y)/np.sum(x)
    return p/N


def Recall(X_pre, X_gt):
    N = len(X_pre)
    p = 0.0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += np.sum(x * y) / np.sum(y)
    return p/N


def F1(X_pre, X_gt):
    N = len(X_pre)
    p = 0
    for i in range(N):
        x = X_pre[i, :]
        y = X_gt[i, :]
        p += 2*np.sum(x * y) / (np.sum(x) + np.sum(y))
    return p/N

def event_level(SO_av, GT_av):
    # extract events
    N = SO_av.shape[0]
    event_p_av = [None for n in range(N)]
    event_gt_av = [None for n in range(N)]

    TP_av = np.zeros(N)
    FP_av = np.zeros(N)
    FN_av = np.zeros(N)

    for n in range(N):
        seq_pred = SO_av[n, :]
        if np.sum(seq_pred) != 0:
            x = extract_event(seq_pred, n)
            event_p_av[n] = x

        seq_gt = GT_av[n, :]
        if np.sum(seq_gt) != 0:
            x = extract_event(seq_gt, n)
            event_gt_av[n] = x

        tp, fp, fn = event_wise_metric(event_p_av[n], event_gt_av[n])
        TP_av[n] += tp
        FP_av[n] += fp
        FN_av[n] += fn
        
    # n = len(FP_av)
    F_av = []
    for ii in range(N):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))

    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))
    # pdb.set_trace()

    return f_av


def segment_level(SO_av, GT_av):
    # SO_av, GT_av: [46/67+1, 10]
    # compute F scores
    TP_av = np.sum(SO_av * GT_av, axis=1)
    FN_av = np.sum((1 - SO_av) * GT_av, axis=1)
    FP_av = np.sum(SO_av * (1 - GT_av), axis=1)
    n = len(FP_av)
    # pdb.set_trace()
    F_av = []
    for ii in range(n):
        if (TP_av + FP_av)[ii] != 0 or (TP_av + FN_av)[ii] != 0:
            F_av.append(2 * TP_av[ii] / (2 * TP_av[ii] + (FN_av + FP_av)[ii]))
    # pdb.set_trace()

    if len(F_av) == 0:
        f_av = 1.0 # all true negatives
    else:
        f_av = (sum(F_av)/len(F_av))
    # pdb.set_trace()

    return f_av


def to_vec(start, end):
    x = np.zeros(10)
    for i in range(start, end):
        x[i] = 1
    return x

def extract_event(seq, n):
    x = []
    i = 0
    while i < 10:
        if seq[i] == 1:
            start = i
            if i + 1 == 10:
                i = i + 1
                end = i
                x.append(to_vec(start, end))
                break

            for j in range(i + 1, 10):
                if seq[j] != 1:
                    i = j + 1
                    end = j
                    x.append(to_vec(start, end))
                    break
                else:
                    i = j + 1
                    if i == 10:
                        end = i
                        x.append(to_vec(start, end))
                        break
        else:
            i += 1
    return x

def event_wise_metric(event_p, event_gt):
    TP = 0
    FP = 0
    FN = 0

    if event_p is not None:
        num_event = len(event_p)
        for i in range(num_event):
            x1 = event_p[i]
            if event_gt is not None:
                nn = len(event_gt)
                flag = True
                for j in range(nn):
                    x2 = event_gt[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2): #0.5
                        TP += 1
                        flag = False
                        break
                if flag:
                    FP += 1
            else:
                FP += 1
        # pdb.set_trace()

    if event_gt is not None:
        num_event = len(event_gt)
        for i in range(num_event):
            x1 = event_gt[i]
            if event_p is not None:
                nn = len(event_p)
                flag = True
                for j in range(nn):
                    x2 = event_p[j]
                    if np.sum(x1 * x2) >= 0.5 * np.sum(x1 + x2 - x1 * x2): #0.5
                        flag = False
                        break
                if flag:
                    FN += 1
            else:
                FN += 1
        # pdb.set_trace()
    return TP, FP, FN