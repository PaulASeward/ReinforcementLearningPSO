
import tensorflow as tf
import csv
import math


class CEC_functions:
    def __init__(self, dim):
        csv_file = open('extdata/M_D' + str(dim) + '.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        csv_data_not_null = [[float(data) for data in row if len(data) > 0] for row in csv_data]
        self.rotate_data = tf.constant(csv_data_not_null, dtype=tf.float32)
        csv_file = open('extdata/shift_data.txt')
        csv_data = csv.reader(csv_file, delimiter=' ')
        self.sd = []
        for row in csv_data:
            self.sd += [float(data) for data in row if len(data) > 0]
        self.M1 = self.read_M(dim, 0)
        self.M2 = self.read_M(dim, 1)
        self.O = self.shift_data(dim, 0)
        self.aux9_1 = tf.constant([0.5 ** j for j in range(0, 21)], dtype=tf.float32)
        self.aux9_2 = tf.constant([3 ** j for j in range(0, 21)], dtype=tf.float32)
        self.aux16 = tf.constant([2 ** j for j in range(1, 33)], dtype=tf.float32)

    def read_M(self, dim, m):
        return self.rotate_data[m * dim: (m + 1) * dim]

    def shift_data(self, dim, m):
        return tf.constant(self.sd[m * dim: (m + 1) * dim], dtype=tf.float32)

    def carat(self, dim, alpha):
        return alpha ** (tf.range(dim, dtype=tf.float32) / (2 * (dim - 1)))

    def T_asy(self, X, Y, beta):
        D = len(X)
        for i in range(D):
            if X[i] > 0:
                Y[i] = X[i] ** (1 + beta * (i / (D - 1)) * tf.sqrt(X[i]))
        pass

    def T_osz(self, X):
        for i in [0, -1]:
            c1 = 10 if X[i] > 0 else 5.5
            c2 = 7.9 if X[i] > 0 else 3.1
            x_hat = 0 if X[i] == 0 else tf.math.log(tf.math.abs(X[i]))
            X[i] = tf.math.sign(X[i]) * tf.math.exp(x_hat + 0.049 * (tf.math.sin(c1 * x_hat) + tf.math.sin(c2 * x_hat)))
        pass

    def cf_cal(self, X, delta, bias, fit):
        d = len(X)
        W_star = []
        cf_num = len(fit)

        for i in range(cf_num):
            X_shift = X - self.shift_data(d, i)
            W = 1 / tf.sqrt(tf.reduce_sum(X_shift ** 2)) * tf.exp(-1 * tf.reduce_sum(X_shift ** 2) / (2 * d * delta[i] ** 2))
            W_star.append(W)

        if (tf.reduce_max(W_star) == 0):
            W_star = [1] * cf_num

        omega = W_star / tf.reduce_sum(W_star) * (fit + bias)

        return tf.reduce_sum(omega)

    def Y_matrix(self, X, fun_num, rflag=None):
        result = tf.map_fn(lambda row: self.Y(row, fun_num, rflag), X)
        return result

    def Y(self, X, fun_num, rflag=None):
        import tensorflow as tf

        if rflag is None:
            rf = [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1][fun_num - 1]
        else:
            rf = rflag

        # Unimodal Functions
        # Sphere Function
        # 1
        if fun_num == 1:
            Z = X - self.O
            if rf == 1:
                Z = tf.matmul(self.M1, Z)
            Y = tf.reduce_sum(Z ** 2) - 1400

        # Rotated High Conditioned Elliptic Function
        # 2
        elif fun_num == 2:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_osz(X_rotate)
            Y = tf.reduce_sum((1e6 ** (tf.range(d, dtype=tf.float32) / (d - 1))) * X_rotate ** 2) - 1300


        # Rotated Bent Cigar Function
        # 3
        elif fun_num == 3:
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = tf.matmul(self.M2, X_shift)

            Y = Z[0] ** 2 + 1e6 * tf.reduce_sum(Z[1:] ** 2) - 1200

        # Rotated Discus Function
        # 4
        elif fun_num == 4:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_osz(X_rotate)
            Y = (1e6) * (X_rotate[0] ** 2) + tf.reduce_sum(X_rotate[1:d] ** 2) - 1100

        # Different Powers Function
        # 5
        elif fun_num == 5:
            d = X.shape[0]
            Z = X - self.O
            if rf == 1:
                Z = tf.matmul(self.M1, Z)
            exponents = 2 + (4 * tf.range(d, dtype=tf.float32) / (d - 1))
            Y = tf.sqrt(tf.reduce_sum(tf.abs(Z) ** exponents)) - 1000

        # Basic Multimodal Functions
        # Rotated Rosenbrock's Function
        # 6
        elif fun_num == 6:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, (2.048 * X_shift) / 100)
            Z = X_rotate + 1
            Y = tf.reduce_sum(100 * (Z[:d - 1] ** 2 - Z[1:d]) ** 2 + (Z[:d - 1] - 1) ** 2) - 900


        # Rotated Ackley's Function
        # 7
        elif fun_num == 7:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = tf.matmul(self.M2, (
                        self.carat(d, 10) * X_shift))

            Z = tf.sqrt(Z[:-1] ** 2 + Z[1:] ** 2)
            Y = (tf.reduce_sum(tf.sqrt(Z) + tf.sqrt(Z) * tf.sin(50 * Z ** 0.2) ** 2) / (
                        d - 1)) ** 2 - 800


        # Rotated Weierstrass Function
        # 8
        elif fun_num == 8:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = tf.matmul(self.M2, (
                        self.carat(d, 10) * X_shift))

            Y = -20 * tf.exp(-0.2 * tf.sqrt(tf.reduce_sum(Z ** 2) / d)) \
                - tf.exp(tf.reduce_sum(tf.cos(2 * tf.constant(math.pi) * Z)) / d) \
                + 20 + tf.exp(1) - 700

        # Rotated Weierstrass Function
        # 9
        elif fun_num == 9:
            d = X.shape[0]
            X_shift = 0.005 * (X - self.O)
            X_rotate_1 = tf.matmul(self.M1, X_shift)
            self.T_asy(X_rotate_1, X_shift, 0.5)
            Z = tf.matmul(self.M2, (
                        self.carat(d, 10) * X_shift))

            # kmax = 20
            def _w(v):
                return tf.reduce_sum(self.aux9_1 * tf.cos(2.0 * tf.constant(math.pi) * self.aux9_2 * v))

            Y = tf.reduce_sum([_w(Z[i] + 0.5) for i in range(d)]) - (
                        _w(0.5) * d) - 600


        # Rotated Griewank’s Function
        # 10
        elif fun_num == 10:
            d = X.shape[0]
            X_shift = (600.0 * (X - self.O)) / 100.0
            X_rotate = tf.matmul(self.M1, X_shift)
            Z = self.carat(d, 100) * X_rotate

            Y = 1.0 + (tf.reduce_sum(Z ** 2) / 4000.0) - tf.reduce_prod(tf.cos(Z / tf.sqrt(
                tf.range(1, d + 1, dtype=tf.float64)))) - 500
            import tensorflow as tf

        # Rastrigin’s Function
        # 11
        elif fun_num == 11:
            d = X.shape[0]
            X_shift = 0.0512 * (X - self.O)
            if rf == 1:
                X_shift = tf.matmul(self.M1, X_shift)
            X_osz = tf.identity(X_shift)
            self.T_osz(X_osz)
            self.T_asy(X_osz, X_shift, 0.2)
            if rf == 1:
                X_shift = tf.matmul(self.M2, X_shift)
            Z = self.carat(d, 10) * X_shift

            Y = tf.reduce_sum(10 + Z ** 2 - 10 * tf.cos(
                2 * tf.constant(math.pi) * Z)) - 400

        # Rotated Rastrigin’s Function
        # 12
        elif fun_num == 12:
            d = X.shape[0]
            X_shift = 0.0512 * (X - self.O)
            X_rotate = tf.matmul(self.M1, X_shift)
            X_hat = tf.identity(X_rotate)
            self.T_osz(X_hat)
            self.T_asy(X_hat, X_rotate, 0.2)
            X_rotate = tf.matmul(self.M2, X_rotate)
            X_rotate = self.carat(d, 10) * X_rotate
            Z = tf.matmul(self.M1, X_rotate)

            Y = tf.reduce_sum(10 + Z ** 2 - 10 * tf.cos(
                2 * tf.constant(math.pi) * Z)) - 300  # Expression evaluation using TensorFlow operations

        # Non-continuous Rotated Rastrigin’s Function
        # 13
        elif fun_num == 13:
            d = X.shape[0]
            X_shift = 0.0512 * (X - self.O)
            X_hat = tf.matmul(self.M1, X_shift)


            X_hat = tf.where(tf.abs(X_hat) > 0.5, tf.round(X_hat * 2.0) / 2.0, X_hat)

            Y_hat = tf.identity(X_hat)
            self.T_osz(Y_hat)
            self.T_asy(Y_hat, X_hat, 0.2)
            X_rotate = tf.matmul(self.M2, Y_hat)
            X_carat = self.carat(d, 10) * X_rotate
            Z = tf.matmul(self.M1, X_carat)

            Y = tf.reduce_sum(10 + Z ** 2 - 10 * tf.cos(
                2 * tf.constant(math.pi) * Z)) - 200

            # Schwefel’s Function
        # 14 15
        elif fun_num == 14 or fun_num == 15:
            d = X.shape[0]  # Length of X along the first dimension
            X_shift = 10 * (X - self.O)  # Element-wise calculation with X_shift
            if rf:
                X_shift = tf.matmul(self.M1, X_shift)  # Matrix multiplication similar to M1 @ X_shift
            Z = self.carat(d, 10) * X_shift + 420.9687462275036

            def schwefel_transform(z):
                return tf.where(tf.abs(z) <= 500, z * tf.sin(tf.sqrt(tf.abs(z))),
                                tf.where(z > 500, (500 - z % 500) * tf.sin(tf.sqrt(500 - z % 500)) - (z - 500) ** 2 / (
                                            10000 * d),
                                         (tf.abs(z) % 500 - 500) * tf.sin(tf.sqrt(500 - tf.abs(z) % 500)) - (
                                                     z + 500) ** 2 / (10000 * d)))

            Z = schwefel_transform(Z)

            Y = 418.9828872724338 * d - tf.reduce_sum(Z) + (-100 if fun_num == 14 else 100)

        # Rotated Katsuura Function
        # 16
        elif fun_num == 16:
            d = X.shape[0]
            X_shift = 0.05 * (X - self.O)
            X_rotate = tf.matmul(self.M1, X_shift)
            X_carat = self.carat(d, 100) * X_rotate
            Z = tf.matmul(self.M2, X_carat)

            def _kat(c):
                return tf.reduce_sum(tf.abs(self.aux16 * c - tf.round(self.aux16 * c)) / self.aux16)

            for i in range(d):
                Z[i] = (1 + (i + 1) * _kat(Z[i]))

            Z = tf.reduce_prod(Z ** (10 / d ** 1.2))
            Y = (10 / d ** 2) * Z - (10 / d ** 2) + 200

        # bi-Rastrigin Function
        # 17 18
        elif fun_num == 17 or fun_num == 18:
            d = X.shape[0]  # Length of X along the first dimension
            mu_0 = 2.5
            S = 1 - 1 / ((2 * tf.sqrt(d + 20)) - 8.2)
            mu_1 = -1 * tf.sqrt((mu_0 ** 2 - 1) / S)
            X_star = self.O
            X_shift = 0.1 * (X - self.O)
            X_hat = 2 * tf.sign(X_star) * X_shift + mu_0

            MU_0 = tf.ones(d) * mu_0
            Z = X_hat - MU_0
            if rf:
                Z = tf.matmul(self.M1, Z)  # Matrix multiplication similar to M1 @ Z
            Z = self.carat(d, 100) * Z
            if rf:
                Z = tf.matmul(self.M2, Z)  # Matrix multiplication similar to M2 @ Z

            Y_1 = (X_hat - mu_0) ** 2
            Y_2 = (X_hat - mu_1) ** 2

            Y_3 = tf.reduce_min(tf.reduce_sum(Y_1), d + S * tf.reduce_sum(Y_2))
            Y = Y_3 + 10 * (d - tf.reduce_sum(tf.cos(2 * tf.constant(math.pi) * Z))) + (
                300 if fun_num == 17 else 400)

        # Rotated Expanded Griewank’s plus Rosenbrock’s Function
        # 19
        elif fun_num == 19:
            d = X.shape[0]  # Length of X along the first dimension
            X_shift = 0.05 * (X - self.O) + 1

            tmp = X_shift ** 2 - tf.roll(X_shift, shift=-1, axis=0)
            tmp = 100 * tmp ** 2 + (X_shift - 1) ** 2
            Z = tf.reduce_sum(tmp ** 2 / 4000 - tf.cos(tmp) + 1)

            Y = Z + 500

        # Rotated Expanded Scaffer’s F6 Function
        # 20
        elif fun_num == 20:
            d = X.shape[0]
            X_shift = X - self.O
            X_rotate = tf.matmul(self.M1, X_shift)
            self.T_asy(X_rotate, X_shift, 0.5)
            Z = tf.matmul(self.M2, X_shift)

            tmp1 = Z ** 2 + tf.roll(Z, shift=-1, axis=0) ** 2

            Y = tf.reduce_sum(0.5 + (tf.sin(tf.sqrt(tmp1)) ** 2 - 0.5) / (
                        1 + 0.001 * tmp1) ** 2) + 600

        # Composition Function 1
        # 21
        elif fun_num == 21:
            d = X.shape[0]  # Length of X along the first dimension
            delta = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)
            bias = tf.constant([0, 100, 200, 300, 400], dtype=tf.float32)
            fit = []

            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 6, rf) + 900) / 1)

            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 5, rf) + 1000) / 1e6)

            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 3, rf) + 1200) / 1e26)

            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 4, rf) + 1100) / 1e6)

            self.O = self.shift_data(d, 5)
            self.M1 = self.read_M(d, 5)
            self.M2 = self.read_M(d, 1)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)

            fit = tf.stack(fit, axis=0)
            Y = self.cf_cal(X, delta, bias, fit) + 700

        # 22
        elif fun_num == 22:

            d = X.shape[0]
            delta = tf.constant([20, 20, 20], dtype=tf.float32)
            bias = tf.constant([0, 100, 200], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 14, rf) + 100) / 1)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 14, rf) + 100) / 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 14, rf) + 100) / 1)
            fit = tf.stack(fit, axis=0)

            Y = self.cf_cal(X, delta, bias, fit) + 800


        # 23
        elif fun_num == 23:
            d = X.shape[0]
            delta = tf.constant([20, 20, 20], dtype=tf.float32)
            bias = tf.constant([0, 100, 200], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) / 1)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 15, rf) - 100) / 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) / 1)
            fit = tf.stack(fit, axis=0)

            Y = self.cf_cal(X, delta, bias, fit) + 900


        # 24
        elif fun_num == 24:
            d = X.shape[0]
            delta = tf.constant([20, 20, 20], dtype=tf.float32)
            bias = tf.constant([0, 100, 200], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)
            fit = tf.stack(fit, axis=0)

            Y = self.cf_cal(X, delta, bias, fit) + 1000


        # 25
        elif fun_num == 25:
            d = X.shape[0]
            delta = tf.constant([10, 30, 50], dtype=tf.float32)
            bias = tf.constant([0, 100, 200], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)
            fit = tf.stack(fit, axis=0)

            Y = self.cf_cal(X, delta, bias, fit) + 1100

        # 26
        elif fun_num == 26:
            d = X.shape[0]  # Length of X along the first dimension
            delta = tf.constant([10, 10, 10, 10, 10], dtype=tf.float32)
            bias = tf.constant([0, 100, 200, 300, 400], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 15, rf) - 100) * 0.25)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 1)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 2, rf) + 1300) / 1e7)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 9, rf) + 600) * 2.5)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 10, rf) + 500) * 10)
            fit = tf.stack(fit, axis=0)
            Y = self.cf_cal(X, delta, bias, fit) + 1200

        # 27
        elif fun_num == 27:
            d = X.shape[0]
            delta = tf.constant([10, 10, 10, 20, 20], dtype=tf.float32)
            bias = tf.constant([0, 100, 200, 300, 400], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 10, rf) + 500) * 100)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append((self.Y(X, 12, rf) + 300) * 10)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) * 2.5)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append((self.Y(X, 9, rf) + 600) * 25)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)
            fit = tf.stack(fit, axis=0)
            Y = self.cf_cal(X, delta, bias, fit) + 1300

        # 28
        elif fun_num == 28:
            d = X.shape[0]
            delta = tf.constant([10, 20, 30, 40, 50], dtype=tf.float32)
            bias = tf.constant([0, 100, 200, 300, 400], dtype=tf.float32)
            fit = []
            fit.append((self.Y(X, 19, rf) - 500) * 2.5)
            self.O = self.shift_data(d, 1)
            self.M1 = self.read_M(d, 1)
            self.M2 = self.read_M(d, 2)
            fit.append(((self.Y(X, 7, rf) + 800) * 2.5) / 1e3)
            self.O = self.shift_data(d, 2)
            self.M1 = self.read_M(d, 2)
            self.M2 = self.read_M(d, 3)
            fit.append((self.Y(X, 15, rf) - 100) * 2.5)
            self.O = self.shift_data(d, 3)
            self.M1 = self.read_M(d, 3)
            self.M2 = self.read_M(d, 4)
            fit.append(((self.Y(X, 20, rf) - 600) * 5) / 1e4)
            self.O = self.shift_data(d, 4)
            self.M1 = self.read_M(d, 4)
            self.M2 = self.read_M(d, 5)
            fit.append((self.Y(X, 1, rf) + 1400) / 1e1)
            fit = tf.stack(fit, axis=0)

            Y = self.cf_cal(X, delta, bias, fit) + 1400


        return Y


if __name__ == "__main__":
    f_num = 9
    cec_functions = CEC_functions(30)

    X = tf.ones(30, dtype=tf.float32)

    # C Calculations
    # import cic13functions
    # C_Y = np.longdouble(cic13functions.run(str(f_num) + ',' + str(list(X))[1:-1]))

    # Python Calculations
    P_Y = cec_functions.Y(X, f_num)

    # print('c response:', C_Y )
    print('python response:', P_Y)
    pass
