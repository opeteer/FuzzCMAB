package id.ac.skripsi.fuzzcmab.util;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Utilitas Aljabar Linier Murni untuk Regresi Ridge Kontekstual CMAB (d = 10).
 *
 * Menyediakan operasi vektor dan matriks presisi tinggi (double / 64-bit IEEE 754)
 * tanpa ketergantungan pustaka eksternal:
 *   - Perkalian titik (dot product), hasil kali luar (outer product x * x^T)
 *   - Perkalian matriks-vektor (A * x) dan matriks-matriks (A * B)
 *   - Bentuk kuadratik Mahalanobis (x^T * A^{-1} * x) dan bentuk sandwich (x^T * V^{-1} * V_tilde * V^{-1} * x)
 *   - Inversi matriks Gauss-Jordan dengan partial pivoting: A^{-1}
 *   - Pembaruan invers rank-1 Sherman-Morrison O(d^2)
 */
public final class LinearAlgebra {

    private LinearAlgebra() {
    }

    public static double clip(double val, double minVal, double maxVal) {
        return Math.max(minVal, Math.min(maxVal, val));
    }

    public static double[] copyVector(double[] v) {
        return Arrays.copyOf(v, v.length);
    }

    public static double[][] copyMatrix(double[][] m) {
        double[][] out = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            out[i] = Arrays.copyOf(m[i], m[i].length);
        }
        return out;
    }

    public static double[][] identity(int d, double scale) {
        double[][] eye = new double[d][d];
        for (int i = 0; i < d; i++) {
            eye[i][i] = scale;
        }
        return eye;
    }

    public static double dot(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Dimensi dot product tidak cocok: " + a.length + " vs " + b.length);
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public static double l2Norm(double[] v) {
        return Math.sqrt(Math.max(0.0, dot(v, v)));
    }

    public static double[][] outer(double[] x, double[] y) {
        int m = x.length;
        int n = y.length;
        double[][] out = new double[m][n];
        for (int i = 0; i < m; i++) {
            double xi = x[i];
            for (int j = 0; j < n; j++) {
                out[i][j] = xi * y[j];
            }
        }
        return out;
    }

    public static double[] matVec(double[][] a, double[] x) {
        int m = a.length;
        int n = x.length;
        double[] out = new double[m];
        for (int i = 0; i < m; i++) {
            double s = 0.0;
            double[] row = a[i];
            for (int j = 0; j < n; j++) {
                s += row[j] * x[j];
            }
            out[i] = s;
        }
        return out;
    }

    /**
     * Menghitung bentuk kuadratik Mahalanobis x^T * A_inv * x.
     */
    public static double quadraticForm(double[][] aInv, double[] x) {
        double[] ax = matVec(aInv, x);
        return Math.max(0.0, dot(x, ax));
    }

    /**
     * Menghitung bentuk sandwich Russac et al. (2019): x^T * V_inv * V_tilde * V_inv * x.
     */
    public static double sandwichQuadraticForm(double[][] vInv, double[][] vTilde, double[] x) {
        double[] u = matVec(vInv, x);
        double[] vTildeU = matVec(vTilde, u);
        return Math.max(0.0, dot(u, vTildeU));
    }

    /**
     * Pembaruan kovariansi terdiskonto sesuai Persamaan (3.19):
     *     A_{t+1, a} = gamma * A_{t, a} + x_t * x_t^T + (1 - gamma) * lambda_reg * I_d
     */
    public static void updateDiscountedCovarianceInPlace(
            double[][] aMat,
            double[] x,
            double gamma,
            double l2Lambda
    ) {
        int d = x.length;
        double regFloor = (1.0 - gamma) * l2Lambda;
        for (int i = 0; i < d; i++) {
            double xi = x[i];
            for (int j = 0; j < d; j++) {
                aMat[i][j] = gamma * aMat[i][j] + (xi * x[j]);
            }
            aMat[i][i] += regFloor;
        }
    }

    /**
     * Peluruhan satu lengan tanpa observasi baru (ketika decayAllArms = true):
     *     A <- gamma * A + (1 - gamma) * lambda_reg * I_d
     */
    public static void decayCovarianceInPlace(double[][] aMat, double gamma, double l2Lambda) {
        int d = aMat.length;
        double regFloor = (1.0 - gamma) * l2Lambda;
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                aMat[i][j] *= gamma;
            }
            aMat[i][i] += regFloor;
        }
    }

    /**
     * Pembaruan vektor respon terdiskonto sesuai Persamaan (3.20):
     *     b_{t+1, a} = gamma * b_{t, a} + y_t * x_t
     */
    public static void updateDiscountedResponseInPlace(
            double[] bVec,
            double[] x,
            double gamma,
            double scalarFeedback
    ) {
        for (int i = 0; i < bVec.length; i++) {
            bVec[i] = gamma * bVec[i] + scalarFeedback * x[i];
        }
    }

    /**
     * Pembaruan invers Sherman-Morrison rank-1 O(d^2) untuk kasus stasioner (gamma = 1.0):
     *     (A + x x^T)^{-1} = A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
     */
    public static void shermanMorrisonUpdateInPlace(double[][] aInv, double[] x) {
        int d = x.length;
        double[] aInvX = matVec(aInv, x);
        double denom = 1.0 + dot(x, aInvX);
        if (Math.abs(denom) < 1e-12) {
            return;
        }
        for (int i = 0; i < d; i++) {
            double u = aInvX[i];
            for (int j = 0; j < d; j++) {
                aInv[i][j] -= (u * aInvX[j]) / denom;
            }
        }
    }

    /**
     * Inversi matriks persegi d x d menggunakan eliminasi Gauss-Jordan dengan partial pivoting.
     */
    public static double[][] invert(double[][] matrix) {
        int n = matrix.length;
        double[][] aug = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(matrix[i], 0, aug[i], 0, n);
            aug[i][n + i] = 1.0;
        }

        for (int col = 0; col < n; col++) {
            int pivotRow = col;
            double maxVal = Math.abs(aug[col][col]);
            for (int row = col + 1; row < n; row++) {
                double v = Math.abs(aug[row][col]);
                if (v > maxVal) {
                    maxVal = v;
                    pivotRow = row;
                }
            }

            if (pivotRow != col) {
                double[] tmp = aug[col];
                aug[col] = aug[pivotRow];
                aug[pivotRow] = tmp;
            }

            double pivot = aug[col][col];
            if (Math.abs(pivot) < 1e-14) {
                pivot = (pivot >= 0.0 ? 1e-14 : -1e-14);
            }

            for (int j = col; j < 2 * n; j++) {
                aug[col][j] /= pivot;
            }

            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = aug[row][col];
                    if (Math.abs(factor) > 0.0) {
                        for (int j = col; j < 2 * n; j++) {
                            aug[row][j] -= factor * aug[col][j];
                        }
                    }
                }
            }
        }

        double[][] inv = new double[n][n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(aug[i], n, inv[i], 0, n);
        }
        return inv;
    }

    public static int chooseRandom(List<Integer> items, Random rng) {
        if (items.isEmpty()) {
            return 0;
        }
        if (items.size() == 1) {
            return items.get(0);
        }
        return items.get(rng.nextInt(items.size()));
    }
}
