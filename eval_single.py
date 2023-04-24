
import evaluation


if dataset == 'f30k':
    evaluation.evalrank('/model_best.pth.tar',
                        data_path='',
                        split='test', save_path='')
else:
    evaluation.evalrank('/model_best.pth.tar',
                        data_path='',
                        split='testall', fold5=True)
    evaluation.evalrank('/model_best.pth.tar',
                        data_path='',
                        split='testall', fold5=False)
    evaluation.evalrank_eccv('/model_best.pth.tar',
                             data_path='',
                             split='test')
