from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu, sigma = 10, 1
    samples = np.random.normal(mu, sigma, 1000)

    # Question 1 - Draw samples and print fitted model
    uni_gaus = UnivariateGaussian()
    uni_gaus.fit(samples)

    print(f'({uni_gaus.mu_}, {uni_gaus.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    estimations_error = [np.abs(uni_gaus.fit(samples[:sample_size]).mu_ - mu) for sample_size in range(10, 1001, 10)]

    estimations_df = pd.DataFrame(np.array([np.arange(10, 1001, 10), np.array(estimations_error)]).transpose(),
                                  columns=["smpl_size", "est_err"])

    figure = px.line(estimations_df, x="smpl_size", y="est_err",
                     labels={"smpl_size": "Sample Size", "est_err": "Estimation Error (Distance from real MU)"},
                     title="Estimation Error as a function of Sample Size")
    figure.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_values = uni_gaus.pdf(samples)
    pdf_df = pd.DataFrame(np.array([samples, pdf_values]).transpose(), columns=["smpl", "pdf"])

    figure = px.scatter(pdf_df, x="smpl", y="pdf",
                        labels={"smpl": "Sample Value", "pdf": "Calculated PDF Value"},
                        title="PDF Values of a set of 1000 samples distributed Normal(10, 1)")
    figure.show()


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(mu, cov, 1000)

    # Question 4 - Draw samples and print fitted model
    mult_gaus = MultivariateGaussian()
    mult_gaus.fit(samples)

    print(f'Estimated Expectation: {mult_gaus.mu_}')
    print(f'Estimated Covariance Matrix: {mult_gaus.cov_}')


    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    for i in range(200):
        mu = np.array([f1[i], 0, f3[i], 0])

        print(mult_gaus.log_likelihood(mu, cov, samples))

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
