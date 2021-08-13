import editdistance


def cer(r, h):
    return err(r.strip(), h.strip())


def err(r, h):
    dis = editdistance.eval(r, h)
    if len(r) == 0.0:
        return len(h)

    return float(dis) / float(len(r))


def wer(r, h):
    r = r.split()
    h = h.split()

    return err(r, h)
