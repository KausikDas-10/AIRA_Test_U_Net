class dice_loss:

    def __init__(self, smooth = None):
        self.smooth = smooth
        if self.smooth is None:
            self.smooth = 1.

    def loss(self, pred, target):
        pred = pred.contiguous()
        target = target.contiguous()    

        intersection = (pred * target).sum(dim=2).sum(dim=2)
        
        loss = (1 - ((2. * intersection + self.smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth)))
        
        return loss.mean() 