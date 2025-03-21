# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.stats import multivariate_normal as mvn
from scipy.stats import gamma
import os

np.random.seed(0)  # For reproducibility


class BPCA(object):

    def __init__(self, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3):
        # hyperparameters
        self.a_alpha = a_alpha # parameter of alpha's prior (a Gamma distribution)
        self.b_alpha = b_alpha # parameter of alpha's prior (a Gamma distribution)
        self.a_tau = a_tau     # parameter of tau's prior (a Gamma distribution)
        self.b_tau = b_tau     # parameter of tau's prior (a Gamma distribution)
        self.beta = beta
        # history of ELBOS
        self.elbos = None
        self.variations = None
        # history of log likelihoods
        self.loglikelihoods = None


    def update(
        self,
        fix_tau_at = None
        ):
        """fixed-point update of the Bayesian PCA"""
        # inverse of the sigma^2
        self.tau = self.a_tau_tilde / self.b_tau_tilde
        # hyperparameters controlling the magnitudes of each column of the weight matrix
        self.alpha = self.a_alpha_tilde / self.b_alpha_tilde
        # covariance matrix of the latent variables
        self.cov_z = np.linalg.inv(np.eye(self.q) + self.tau *
                        (np.trace(self.cov_w) + np.dot(self.mean_w.T, self.mean_w)))
        # mean of the latent variable
        self.mean_z = self.tau * np.dot(np.dot(self.cov_z, self.mean_w.T), self.Xb - self.mean_mu)
        # covariance matrix of the mean observation
        self.cov_mu = np.eye(self.d) / (self.beta + self.b * self.tau)
        # mean of the mean observation
        self.mean_mu = self.tau * np.dot(self.cov_mu, np.sum(self.Xb-np.dot(self.mean_w,
                        self.mean_z), axis=1)).reshape(self.d, 1)
        # covariance matrix of each column of the weight matrix
        self.cov_w = np.linalg.inv(np.diag(self.alpha) + self.tau *
                        (self.b * self.cov_z + np.dot(self.mean_z, self.mean_z.T)))
        # mean of each column of the weight matrix
        self.mean_w = self.tau * np.dot(self.cov_w, np.dot(self.mean_z, (self.Xb-self.mean_mu).T)).T
        # estimation of the b in alpha's Gamma distribution
        self.b_alpha_tilde = self.b_alpha + 0.5 * (np.trace(self.cov_w) +
                        np.diag(np.dot(self.mean_w.T, self.mean_w)))
        # estimation of the b in tau's Gamma distribution
        if fix_tau_at is None:
            self.b_tau_tilde = self.b_tau + 0.5 * np.trace(np.dot(self.Xb.T, self.Xb)) + \
                            0.5 * self.b*(np.trace(self.cov_mu)+np.dot(self.mean_mu.flatten(), self.mean_mu.flatten())) + \
                            0.5 * np.trace(np.dot(np.trace(self.cov_w)+np.dot(self.mean_w.T, self.mean_w),
                                            self.b*self.cov_z+np.dot(self.mean_z, self.mean_z.T))) + \
                            np.sum(np.dot(np.dot(self.mean_mu.flatten(), self.mean_w), self.mean_z)) + \
                            -np.trace(np.dot(self.Xb.T, np.dot(self.mean_w, self.mean_z))) + \
                            -np.sum(np.dot(self.Xb.T, self.mean_mu))
        else:
            self.b_tau_tilde = self.a_tau_tilde / fix_tau_at
        

    def calculate_log_likelihood(self):
        """calculate the log likelihood of observing self.X"""
        w = self.mean_w
        c = np.eye(self.d)*self.tau + np.dot(w, w.T) 
        xc = self.X - self.X.mean(axis=1).reshape(-1,1)
        s = np.dot(xc, xc.T) / self.N
        self.s = s
        c_inv_s = scipy.linalg.lstsq(c, s)[0]
        loglikelihood = -0.5*self.N*(self.d*np.log(2*np.pi)+np.log(np.linalg.det(c))+np.trace(c_inv_s))
        return loglikelihood


    def calculate_ELBO(self):
        '''ELBO = E_q[-log(q(theta))+log(p(theta)+log(p(Y|theta,X)))]
                = -entropy + logprior + loglikelihood '''

        # random sample
        z = np.array([np.random.multivariate_normal(self.mean_z[:,i], self.cov_z) for i in range(self.b)]).T
        mu = np.random.multivariate_normal(self.mean_mu.flatten(), self.cov_mu)
        w = np.array([np.random.multivariate_normal(self.mean_w[i], self.cov_w) for i in range(self.d)])
        alpha = np.random.gamma(self.a_alpha_tilde, 1/self.b_alpha_tilde)
        tau = np.random.gamma(self.a_tau_tilde, 1/self.b_tau_tilde)

        # entropy
        # q(z)
        entropy = np.sum(np.array([mvn.logpdf(z[:,i], self.mean_z[:,i], self.cov_z) for i in range(self.b)]))

        # q(mu)
        entropy += mvn.logpdf(mu, self.mean_mu.flatten(), self.cov_mu)

        # q(W)
        entropy += np.sum(np.array([mvn.logpdf(w[i], self.mean_w[i], self.cov_w) for i in range(self.d)]))

        # q(alpha)
        entropy += np.sum(gamma.logpdf(alpha, self.a_alpha_tilde, scale=1/self.b_alpha_tilde))

        # q(tau)
        entropy += gamma.logpdf(tau, self.a_tau_tilde, scale=1/self.b_tau_tilde)

        # logprior
        # p(z), z ~ N(0, I)
        logprior = np.sum(np.array([mvn.logpdf(z[:,i], mean=np.zeros(self.q), cov=np.eye(self.q)) for i in range(self.b)]))

        # p(w|alpha), conditional gaussian
        logprior += np.sum(np.array([self.d/2*np.log(alpha[i]/(2*np.pi))-alpha[i]*np.sum(w[:,i]**2)/2 for i in range(self.q)]))

        # p(alpha), alpha[i] ~ Gamma(a, b)
        logprior += np.sum(gamma.logpdf(alpha, self.a_alpha, scale=1/self.b_alpha))

        # p(mu), mu ~ N(0, I/beta)
        logprior += mvn.logpdf(mu, mean=np.zeros(self.d), cov=np.eye(self.d)/self.beta)

        # p(tau), tau ~ Gamma(c, d)
        logprior += gamma.logpdf(tau, self.a_tau, scale=1/self.b_tau)

        # loglikelihood
        pred = np.dot(w, z) + mu.reshape(-1,1)
        loglikelihood = np.sum(np.array([mvn.logpdf(self.Xb[:,i], pred[:,i], np.eye(self.d)/tau) for i in range(self.b)]))

        return -entropy + logprior + loglikelihood


    def batch_idx(self, i):
        if self.b == self.N:
            return np.arange(self.N)
        idx1 = (i*self.b) % self.N
        idx2 = ((i+1)*self.b) % self.N
        if idx2 < idx1:
            idx1 -= self.N
        return np.arange(idx1, idx2)


    def fit(
        self, X=None, batch_size=128, iters=500, print_every=100, verbose=False, trace_elbo=False, trace_loglikelihood=False,
        threshold_alpha_complete = None,
        true_signal_dim = None,
        fix_tau_at= None,
        ):
        """fit the Bayesian PCA model using fixed-point update"""
         # data, # of samples, dims
        self.X = X.T # don't need to transpose X when passing it
        self.d = self.X.shape[0]
        self.N = self.X.shape[1]
        self.q = self.d-1
        self.ed = []
        self.b = min(batch_size, self.N)

        # variational parameters
        self.mean_z = np.random.randn(self.q, self.b) # latent variable
        self.cov_z = np.eye(self.q)
        self.mean_mu = np.random.randn(self.d, 1)
        self.cov_mu = np.eye(self.d)
        self.mean_w = np.random.randn(self.d, self.q)
        self.cov_w = np.eye(self.q)
        self.a_alpha_tilde = self.a_alpha + self.d/2
        self.b_alpha_tilde = np.abs(np.random.randn(self.q))
        self.a_tau_tilde = self.a_tau + self.b * self.d / 2
        self.b_tau_tilde = np.abs(np.random.randn(1))

        # update
        order = np.arange(self.N)
        elbos = np.zeros(iters)
        loglikelihoods = np.zeros(iters)
        self.iter_converge = iters
        for i in range(iters):
            idx = order[self.batch_idx(i)]
            self.Xb = self.X[:,idx]
            self.update(
                fix_tau_at=fix_tau_at
            )
            if (threshold_alpha_complete is not None) and ( true_signal_dim is not None ):
                alpha_sorted = sorted(self.alpha)
                if (alpha_sorted[true_signal_dim] / alpha_sorted[true_signal_dim-1]) > threshold_alpha_complete:
                    self.iter_converge = i
                    break
            if trace_elbo:
                elbos[i] = self.calculate_ELBO()
            if trace_loglikelihood:
                loglikelihoods[i] = self.calculate_log_likelihood()
            if verbose and i % print_every == 0:
                print('Iter %d, LL: %f, alpha: %s' % (i, loglikelihoods[i], str(self.alpha)))
        self.captured_dims()
        self.elbos = elbos if trace_elbo else None
        self.loglikelihoods = loglikelihoods if trace_loglikelihood else None


    def captured_dims(self):
        """return the number of captured dimensions"""
        sum_alpha = np.sum(1/self.alpha)
        self.ed = np.array([i for i, inv_alpha in enumerate(1/self.alpha) if inv_alpha < sum_alpha/self.q])


    def transform(self, X=None, full=True):
        """generate samples from the fitted model"""
        X = self.X if X is None else X.T
        if full:
            w = self.mean_w
            l = self.q
        else:
            w = self.mean_w[:,ed]
            l = len(self.ed)
        m = np.eye(l)*self.tau + np.dot(w.T, w)
        inv_m = np.linalg.inv(m)
        z = np.dot(np.dot(inv_m, w.T), X - self.mean_mu)
        return z.T
        # return np.array([np.random.multivariate_normal(z[:,i], inv_m*self.tau) for i in range(X.shape[1])])


    def inverse_transform(self, z, full=True):
        """transform the latent variable into observations"""
        z = z.T
        if full:
            w = self.mean_w
        else:
            w = self.mean_w[:,ed]
        x = np.dot(w, z) + self.mean_mu
        return x.T


    def fit_transform(self, X=None, batch_size=128, iters=500, print_every=100, verbose=False, trace_elbo=False, trace_loglikelihood=False):
        self.fit(X, batch_size, iters, print_every, verbose, trace_elbo)
        return self.transform()


    def generate(self, size=1):
        """generate samples from the fitted model"""
        w = self.mean_w[:, self.ed]
        c = np.eye(self.d)*self.tau + np.dot(w, w.T)
        return np.array([np.random.multivariate_normal(self.mean_mu.flatten(), c) for i in range(size)])


    def get_weight_matrix(self):
        return self.mean_w


    def get_inv_variance(self):
        return self.alpha


    def get_effective_dims(self):
        return len(self.ed)


    def get_cov_mat(self):
        w = self.mean_w[:, self.ed]
        c = np.eye(self.d)*self.tau + np.dot(w, w.T) 
        return c


    def get_elbo(self):
        return self.elbos


    def get_loglikelihood(self):
        return self.loglikelihoods


def simulate_data(psi=1, N=100, P=10):
    psi_inv = 1 / psi
    cov = np.diag([5, 4, 3, 2] + [psi_inv] * (P - 4))

    return np.random.multivariate_normal(np.zeros(P), cov, N)

# %%

# %%
from scipy.stats import multivariate_normal as mvn, gamma

class GibbsBayesianPCA:
    def __init__(
        self, t, q, a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3,
        tau_init=None, 
        ):
        self.t = t
        self.N, self.d = t.shape
        self.q = q
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta
        self.initialize_parameters(tau_init=tau_init)

    def initialize_parameters(
        self,
        tau_init=None,
        ):
        self.x = np.random.randn(self.N, self.q)
        self.mu = np.zeros((1, self.d))
        self.W = np.random.randn(self.d, self.q)
        self.alpha = gamma.rvs(self.a_alpha, scale=1/self.b_alpha, size=self.q)
        self.tau = 1 if tau_init is None else tau_init

        self.Iq = np.eye(self.q)
        self.Id = np.eye(self.d)


    def update_x(self):
        Sigma_x = np.linalg.inv(self.Iq + self.tau * self.W.T @ self.W)
        m_x = self.tau * Sigma_x @ self.W.T @ (self.t - self.mu).T
        for n in range(self.N):
            m_x_n = m_x[:, [n]]
            self.x[[n]] = mvn.rvs(mean=m_x_n.flatten(), cov=Sigma_x)

    def update_mu(self):
        Sigma_mu = np.linalg.inv(self.beta * self.Id + self.N * self.tau * self.Id)
        m_mu = self.tau * Sigma_mu @ np.sum(self.t - self.x @ self.W.T, axis=0)
        self.mu = mvn.rvs(mean=m_mu, cov=Sigma_mu)[None, :]

    def update_W(self):
        Sigma_w = np.linalg.inv(np.diag(self.alpha) + self.tau * self.x.T @ self.x)
        m_w = self.tau * Sigma_w @ self.x.T @ (self.t - self.mu)
        for k in range(self.d):
            m_w_k = m_w[:, [k]]
            self.W[k] = mvn.rvs(mean=m_w_k.flatten(), cov=Sigma_w)

    def update_alpha(self):
        a_alpha_tilde = self.a_alpha + 0.5 * self.d
        b_alpha_tilde = self.b_alpha + 0.5 * np.sum(self.W ** 2, axis=0)
        for i in range(self.q):
            # b_alpha_tilde_i = self.b_alpha + 0.5 * np.sum(self.W[:, i] ** 2)
            b_alpha_tilde_i = b_alpha_tilde[i]
            self.alpha[i] = gamma.rvs(a_alpha_tilde, scale=1/b_alpha_tilde_i)

    def update_tau(
        self,
        fix_tau_at=None,
        ):
        a_tau_tilde = self.a_tau + 0.5 * self.N * self.d
        if fix_tau_at is None:
            
            b_tau_tilde = self.b_tau + 0.5 * (
                (self.t**2).sum() + (self.mu**2).sum()*self.N + np.trace(
                    self.W.T @ self.W @ (self.x.T @ self.x)
                ) + 2 * (self.mu @ self.W @ self.x.T).sum() + np.sum([
                    - 2 * self.t[[n]] @ self.W @ self.x[[n]].T 
                    for n in range(self.N)
                ]) - 2 * (self.t @ self.mu.T).sum()
            )
            self.tau = gamma.rvs(a_tau_tilde, scale=1/b_tau_tilde)
        else:
            self.tau = fix_tau_at
            b_tau_tilde = a_tau_tilde / self.tau


    def fit(
        self, 
        # params for the gibbs sampler
        iterations=500,
        burn_in=200,
        thinning=10,
        threshold_alpha_complete=None,
        true_signal_dim=None,
        fix_tau_at=None,
        ):

        # store the samples
        self.samples = {
            'x': [],
            'mu': [],
            'W': [],
            'alpha': [],
            'tau': [],
        }

        for i in range(iterations):
            self.iter_converge = iterations
            self.update_x()
            self.update_mu()
            self.update_W()
            self.update_alpha()
            self.update_tau( fix_tau_at=fix_tau_at )
            if (i + 1) % 100 == 0:
                print('Iteration ', ( i + 1 ))

            if i >= burn_in and i % thinning == 0:
                self.samples['x'].append(self.x)
                self.samples['mu'].append(self.mu)
                self.samples['W'].append(self.W)
                self.samples['alpha'].append(self.alpha)
                self.samples['tau'].append(self.tau)

                if (threshold_alpha_complete is not None) and ( true_signal_dim is not None ):
                    # mean of self.samples['alpha']
                    alpha_sorted = sorted(np.mean(self.samples['alpha'] , axis=0))
                    if (alpha_sorted[true_signal_dim] / alpha_sorted[true_signal_dim-1]) > threshold_alpha_complete:
                        self.iter_converge = i
                        break


# %%
var_noise_list = np.logspace(-5, 1, 30)
threshold_alpha_complete = 1e2
iter_end_list = np.zeros(len(var_noise_list))
n_repeat = 1

# %%
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time

dir_out = './gibbs_vs_vi/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)
path_csv_vi = dir_out + 'vi.csv'
path_fig_vi = dir_out + 'vi.pdf'
n_iter_max_vi = 200000
path_csv_gibbs = dir_out + 'gibbs.csv'
path_fig_gibbs = dir_out + 'gibbs.pdf'
n_iter_max_gibbs = 20000

gibbs_or_vi = True

if gibbs_or_vi:
    path_csv = path_csv_gibbs
    path_fig = path_fig_gibbs
    n_iter_max = n_iter_max_gibbs
else:
    path_csv = path_csv_vi
    path_fig = path_fig_vi
    n_iter_max = n_iter_max_vi

# Function to perform the simulation and fitting
def simulate_and_fit(v, n_repeat, n_iter_max, threshold_alpha_complete, gibbs_or_vi=gibbs_or_vi):
    iter_end_list_i = np.zeros(n_repeat)
    time_list_i = np.zeros(n_repeat)
    for j in range(n_repeat):
        start_time = time.time()
        d = simulate_data(psi=v**(-1), N=100)
        # variational inference
        if gibbs_or_vi:
            bpca = GibbsBayesianPCA(d, q=d.shape[1]-1, tau_init=v**(-1))
            bpca.fit(
                iterations=n_iter_max,
                threshold_alpha_complete=threshold_alpha_complete,
                true_signal_dim=4,
                fix_tau_at=v**(-1),
                # fix_tau_at=(1e-2)**(-1),
                )
        else:
            bpca = BPCA(a_alpha=1e-3, b_alpha=1e-3, a_tau=1e-3, b_tau=1e-3, beta=1e-3)
            bpca.fit(
                d, iters=n_iter_max,
                threshold_alpha_complete=threshold_alpha_complete,
                true_signal_dim=4,
                fix_tau_at=v**(-1),
                # fix_tau_at=(1e-2)**(-1),
            )
        iter_end_list_i[j] = bpca.iter_converge
        time_list_i[j] = time.time() - start_time
    return np.mean(iter_end_list_i), np.mean(time_list_i)


# Specify the number of worker processes (e.g., 4)
num_workers = 30

r_list = []
# Use ProcessPoolExecutor to parallelize the loop
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(simulate_and_fit, v, n_repeat, n_iter_max, threshold_alpha_complete): v for v in var_noise_list}
    for future in as_completed(futures):
        v = futures[future]
        mean_iter, mean_time = future.result()
        r_list.append((v, mean_iter, mean_time))
        print('Completed Variance: ', v)

# sort r_list by variance
r_list = sorted(r_list, key=lambda x: x[0])
v_list, iter_end_list, time_list = zip(*r_list)
# Convert lists to numpy arrays for plotting
iter_end_list = np.array(iter_end_list)
time_list = np.array(time_list)
np.savetxt(path_csv, np.array(r_list), delimiter=',', header='noise_variance,iter_to_complete,seconds_to_complete', comments='')

# %%
r_list_vi = np.loadtxt(path_csv_vi, delimiter=',', skiprows=1)
r_list_gibbs = np.loadtxt(path_csv_gibbs, delimiter=',', skiprows=1)
v_list_gibbs, iter_end_list_gibbs, time_list_gibbs = r_list_gibbs[:, 0], r_list_gibbs[:, 1], r_list_gibbs[:, 2]

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

def plot_one(ax, r_list, n_iter_max, color, method):

    v_list, iter_end_list, time_list = r_list[:, 0], r_list[:, 1], r_list[:, 2]

    # Sort the data points by x-value (log10 of variance)
    x = np.log10(v_list)
    y = time_list
    idtf_mask = iter_end_list < n_iter_max
    sorted_indices = np.argsort(x)
    x = x[sorted_indices]
    y = y[sorted_indices]
    idtf_mask = idtf_mask[sorted_indices]

    # Find transition points between identified and non-identified regions
    transitions = np.where(np.diff(idtf_mask))[0]

    if len(transitions) == 2:
        # First segment (dashed)
        ax.plot(x[:transitions[0]+1], y[:transitions[0]+1], 
                label=method + ' (not identified)', marker='2', linestyle='--', color=color)
        
        # Middle segment (solid)
        ax.plot(x[transitions[0]:transitions[1]+1], y[transitions[0]:transitions[1]+1],
                label=method + ' (identified)', marker='^', linestyle='-', color=color)
        
        # Last segment (dashed)
        ax.plot(x[transitions[1]:], y[transitions[1]:], marker='2', linestyle='--', color=color)
    else:
        # Fallback if there aren't exactly two transition points
        ax.plot(x[idtf_mask], y[idtf_mask], 
                label=method + ' (identified)', marker='2', linestyle='-', color=color)
        ax.plot(x[~idtf_mask], y[~idtf_mask], marker='2', linestyle='--', color=color)

    ax.set_xlabel('Log10 Noise Variance')
    ax.set_ylabel('Time to Identification (seconds)')
    ax.set_title('Time to Identification vs. Noise Variance')
    ax.legend()

plot_one(ax, r_list_vi, n_iter_max_vi, color='blue', method = 'VI')
plot_one(ax, r_list_gibbs, n_iter_max_gibbs, color='red', method = 'MCMC')

both_identified = np.logical_and( r_list_vi[:, 1] < n_iter_max_vi, r_list_gibbs[:, 1] < n_iter_max_gibbs )

assert np.allclose(r_list_vi[:, 0], r_list_gibbs[:, 0]), 'Variance mismatch'
both_identified = np.logical_and( r_list_vi[:, 1] < n_iter_max_vi, r_list_gibbs[:, 1] < n_iter_max_gibbs )
print('Both identified variance: ', r_list_vi[both_identified, 0])
r = r_list_gibbs[both_identified, 2] / r_list_vi[both_identified, 2]
print('Ratio of time to identification (Gibbs/VI): ', r.mean(), '+-', r.std(), 'range: ', r.min(), r.max())

plt.tight_layout()
# save the figure
plt.savefig( dir_out + 'time_to_identification.pdf' )
plt.show()


# %%