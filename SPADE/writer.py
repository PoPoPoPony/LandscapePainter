import os


class Writer():
    def __init__(self, rootPath) -> None:
        self.checkPtFilePathG = f"{rootPath}/CheckPt/Generator"
        self.checkPtFilePathD = f"{rootPath}/CheckPt/Discriminator"
        self.imagePath = f"{rootPath}/images"

        os.makedirs(self.checkPtFilePathG, exist_ok=True)
        os.makedirs(self.checkPtFilePathD, exist_ok=True)
        os.makedirs(self.imagePath, exist_ok=True)


    def writeCheckPt(self, epoch, model, modelType):
        epoch = str(epoch).zfill(3)
        if modelType == 'G':
            fileName = os.path.join(self.checkPtFilePathG, f"epoche{epoch}.pt") 
        elif modelType == 'D':
            fileName = os.path.join(self.checkPtFilePathD, f"epoche{epoch}.pt") 
        model.save(model.state_dict(), fileName)
        print(f"[INFO] Save Epoch {epoch} check point files")


    def writeResult(self, epoch, imageTensor, annoTensor):
        pass