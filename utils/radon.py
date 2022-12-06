import numpy as np
import matplotlib.pyplot as plt
from utils.util import generateLocation, unitDisk

class RadonOperator:
    def __init__(self, imageShape: int, refine: int, ttlNumDtctr: int = 16, ttlNumAngl: int = 8):
        self.imageShape = imageShape
        self.refine = refine
        self.n = int(self.refine / self.imageShape)
        self.ttlNumDtctr = ttlNumDtctr
        self.ttlNumAngl = ttlNumAngl
        self.H = self.generateH()
        self.v, self.s, self.u = np.linalg.svd(self.H, full_matrices=False, hermitian=False)
        # for stable computation, remove the small value of eigen values to make inverse stable
        r = np.min(np.where(self.s < 1e-10))
        self.v = self.v[:, :r]
        self.s = self.s[:r]
        self.u = self.u[:r, :]
        self.Himp = self.generateH_imp()


    def radon(self, object):
        object = self._refineMatrix(object)
        return self.H @ object.reshape(-1, 1)


    def iradon(self, sinogram):
        img = self.Himp @ sinogram.reshape(-1, 1)
        return self._descretImage(img.reshape(self.refine, self.refine))


    def generateH_imp(self):
        return self.u.T @ np.linalg.inv(np.diag(self.s)) @ self.v.T


    def generateH(self):
        coor = generateLocation(self.refine)
        coor = coor.reshape(-1, 2) # (h*w, 2)
        angles = np.arange(self.ttlNumAngl) / self.ttlNumAngl * np.pi
        H = []
        halfStripeWidth = 1 / (self.ttlNumDtctr * 2)
        for angle in list(angles):
            centers = self._generateDetectorCenters(angle)
            end = self._rotateLocation(np.array([0., -1/2]).reshape(-1, 2), angle)
            dotProduct = (coor - end) @ (centers - end).T
            centers_norm = np.linalg.norm(centers - end, ord=2, axis=-1)
            dotProduct = np.round(dotProduct / centers_norm - centers_norm, 10)
            dotProduct = np.logical_and(dotProduct > - halfStripeWidth, dotProduct <= halfStripeWidth).astype(np.float64)
            dotProduct = dotProduct * unitDisk(coor).reshape(-1, 1)
            H.append(dotProduct)
        H = np.round(np.concatenate(H, axis=-1), 0).T
        return H / np.sum(H, axis=-1, keepdims=True)


    def _refineMatrix(self, matrix):
        """
        make the given matrix be more fined so the radon will be more precise
        """
        newMat = np.repeat(matrix, self.n, axis=0)
        newMat = np.repeat(newMat, self.n, axis=1)
        return newMat


    def _descretImage(self, img):
        """
        reduce the img to the targetSize by averaging the block.
        """
        # row
        rows = img[::self.n, :]
        for i in range(1, self.n):
            rows += img[i::self.n, :]
        # cols
        cols = rows[:, ::self.n]
        for i in range(1, self.n):
            cols += rows[:, i::self.n]
        return cols / (self.n**2)


    def _generateDetectorCenters(self, angle):
        """
        Help method for generate H
        """
        stripeWidth = 1 / self.ttlNumDtctr
        x = np.linspace(-1/2 + stripeWidth / 2, 1/2 - stripeWidth / 2, self.ttlNumDtctr)
        y = np.zeros(x.shape)
        centers = np.stack([y, x], axis=-1)
        return self._rotateLocation(centers, angle)


    def _rotateLocation(self, location, angle):
        """
        Help method for generate H
        """
        oriShape = location.shape
        location = location.reshape(-1, 2)
        location = location @ np.array([[ np.cos(angle),-np.sin(angle)],
                                        [ np.sin(angle), np.cos(angle)]])
        location = location.reshape(oriShape)
        return location


    def radon_continuous(self, object, refine, angles=8, detecter=16):
        """
        This is the vision of getting sinogram from resample from continous domain, Not used in this project
        """
        pass
        loc = generateLocation(refine)
        n = refine // detecter
        measurement = []
        for i in range(angles):
            angle = i * (1/angles) * np.pi
            loc_r = self._rotateLocation(loc, angle)
            sig = object.getSignalPresentObject(loc_r)
            sig = unitDisk(loc_r) * sig
            # plt.figure()
            # plt.imshow(sig)
            # plt.show()
            sig = np.mean(sig, axis=0)

            sum = np.zeros((detecter,))
            for j in range(n):
                sum += sig[j::n]
            measurement.append(sum / n)
        return np.stack(measurement, axis = -1)


    def get_s(self):
        return self.s


    def get_u(self):
        return self.u


    def get_v(self):
        return self.v