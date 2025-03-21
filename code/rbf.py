from util import RBFKernel2
import numpy as np
import copy
import logging

# %%
def rbf_inner_product(
        param_rbf1,
        param_rbf2,
    ):  
    '''
    inner product between two radial basis functions
    ---
    param_rbf1: dict, keys=['length_scale', 'center', 'multiplier']
    param_rbf2: dict, keys=['length_scale', 'center', 'multiplier']
    '''
    l1 = param_rbf1['length_scale']
    x1 = param_rbf1['center']
    m1 = param_rbf1['multiplier']

    l2 = param_rbf2['length_scale']
    x2 = param_rbf2['center']
    m2 = param_rbf2['multiplier']

    l_kkp = l1 * l2 / (l1**2 + l2**2)**0.5
    
    m_tmp = ( l1**2 * x2 + l2**2 * x1 )
    m_kkp = np.exp(
        ( m_tmp @ m_tmp.T - (l1**2 + l2**2) * (l1**2 * x2 @ x2.T + l2**2 * x1 @ x1.T ) ) \
            / (2 * l1**2 * l2**2 * (l1**2 + l2**2) )
    )
    return m1 * m2 * m_kkp  * ( 2 * np.pi * l_kkp**2 )**(x1.shape[1]/2)





class RadialBasisFunction:
    def __init__(self, length_scale, center, multiplier):
        '''
        length_scale: float
        center: np.array, shape=(1, d_index_set)
        multiplier: float
        '''
        self.length_scale = length_scale
        self.center = center
        self.multiplier = multiplier
    
    def __call__(self, x):
        '''
        x: np.array, shape=(n, d_index_set)
        '''
        rbfk = RBFKernel2(sigma=0, length_scale=self.length_scale, multiplier=self.multiplier)
        return rbfk(x1=x, x2=self.center)
    
    def inner_product(self, rbf):
        '''
        inner product between two radial basis functions
        ---
        rbf: RadialBasisFunction object
        '''
        l1 = self.length_scale
        x1 = self.center
        m1 = self.multiplier

        l2 = rbf.length_scale
        x2 = rbf.center
        m2 = rbf.multiplier

        if m1 == 0 or m2 == 0:
            return 0
        else:
            l_kkp_log = np.log(l1) + np.log(l2) - 0.5 * np.log(l1**2 + l2**2)
            
            m_tmp = ( l1**2 * x2 + l2**2 * x1 )
            m_kkp = np.exp(
                ( m_tmp @ m_tmp.T - (l1**2 + l2**2) * (l1**2 * x2 @ x2.T + l2**2 * x1 @ x1.T ) ) \
                    / (2 * l1**2 * l2**2 * (l1**2 + l2**2) )
            )
            m_kkp = np.squeeze(m_kkp)

            if m_kkp == 0:
                return 0
            else:
                ip_log = np.log(np.abs(m1)) + np.log(np.abs(m2)) + np.log(m_kkp) + (x1.shape[1]/2) * ( np.log(2 * np.pi) + 2 * l_kkp_log )
                ip = np.sign(m1) * np.sign(m2) * np.exp(ip_log)
                return ip



class RadicalBasisFunctionMixture:
    def __init__(self, radial_basis_function_list):
        '''
        radial_basis_function_list: list of RadialBasisFunction objects
        '''
        self.radial_basis_function_list = radial_basis_function_list
        self.radial_basis_function_mix_list_orthnorm = None
        self.gram_matrix_inv = None
    
    def __call__(self, x):
        return np.sum([rbf(x) for rbf in self.radial_basis_function_list],axis=0)
    
    def inner_product(self, rbfm):
        '''
        inner product between two radial basis function mixtures
        ---
        rbfm: RadicalBasisFunctionMixture object
        '''
        ip = 0
        for rbf_i in self.radial_basis_function_list:
            for rbf_j in rbfm.radial_basis_function_list:
                ip += rbf_i.inner_product(rbf_j)
        return ip


    def get_orthonormal_radial_basis_functions_mix(
            self,
        ):
        '''
        orthogonalize and normalize the radial basis functions
        '''
        if self.radial_basis_function_mix_list_orthnorm is not None:
            # if the list of orthogonalized and normalized radial basis functions has been computed, return it
            return self.radial_basis_function_mix_list_orthnorm
        else:
            radial_basis_function_mix_list_orthnorm = []
            for i, rbf_i in enumerate(self.radial_basis_function_list):
                rbfm_i = RadicalBasisFunctionMixture(radial_basis_function_list=[copy.deepcopy(rbf_i)])
                # initialize the ith orthogonalized and normalized radial basis function mixtures with the first i radial basis functions
                rbfm_i_orthnorm = RadicalBasisFunctionMixture(radial_basis_function_list= copy.deepcopy(self.radial_basis_function_list))

                # compute the combination that is orthogonal to the former mixtures
                for i2, rbfm_i2_orthnorm in enumerate(rbfm_i_orthnorm.radial_basis_function_list):

                    if i2 > i:
                        rbfm_i_orthnorm.radial_basis_function_list[i2].multiplier = 0
                    elif i2 == i:
                        pass
                    else:
                        rbfm_i_orthnorm.radial_basis_function_list[i2].multiplier = - np.sum([
                            # NOTE rbfm_otnm_i3 is of unit length, so the inner product is equal to 1 and can be omitted
                            ( rbfm_i.inner_product( rbfm_otnm_i3 ) ) * rbfm_otnm_i3.radial_basis_function_list[i2].multiplier
                            for i3, rbfm_otnm_i3 in enumerate( radial_basis_function_mix_list_orthnorm )
                        ])

                # normalize the current combination of radial basis functions
                rbfm_i_orthnorm_length = np.exp( 0.5 * np.log( rbfm_i_orthnorm.inner_product(rbfm_i_orthnorm) ) )
                logging.debug(f'rbfm_i_orthnorm_length={rbfm_i_orthnorm_length}')
                for i2, rbfm_i2_orthnorm in enumerate(rbfm_i_orthnorm.radial_basis_function_list):
                    m = rbfm_i_orthnorm.radial_basis_function_list[i2].multiplier
                    if m != 0:
                        rbfm_i_orthnorm.radial_basis_function_list[i2].multiplier = np.sign(m) * np.exp( np.log(np.abs(m)) - np.log(rbfm_i_orthnorm_length) )

                # NOTE the inner product could be off 1 to some extent due to numerical errors
                ip = rbfm_i_orthnorm.inner_product(rbfm_i_orthnorm)
                if np.abs(ip - 1) > 1e-3:
                    logging.warning(f'the basis function is not exactly unit length after normalization, the error is {np.abs(ip - 1)}')
                # # check the inner product is 1
                # ip = rbfm_i_orthnorm.inner_product(rbfm_i_orthnorm)
                # assert np.abs(ip - 1) < 1e-3, f'ip={ip}'

                # check the inner product is 0 with the former orthogonalized and normalized radial basis functions
                for i2, rbfm in enumerate(radial_basis_function_mix_list_orthnorm):
                    ip = np.abs(rbfm_i_orthnorm.inner_product(rbfm))
                    if ip > 1e-3:
                        logging.warning(f'the basis function is not exactly orthogonal to the former basis functions, the error is {ip}')
                    # assert ip < 1e-3, f'ip={ip}'
                
                radial_basis_function_mix_list_orthnorm.append(rbfm_i_orthnorm)
            
            self.radial_basis_function_mix_list_orthnorm = radial_basis_function_mix_list_orthnorm
            return self.radial_basis_function_mix_list_orthnorm

    def project_orthonormal(self, rbfm):
        '''
        project a radial basis function combination onto the subspace spanned by the current list of radial basis functions. this uses the orthonormalized radial basis functions.
        ---
        rbfm: RadicalBasisFunctionMixture object
        ---
        return: RadicalBasisFunctionMixture object
        '''
        rbfm_orthnorm = self.get_orthonormal_radial_basis_functions_mix()
        rbfm_proj = copy.deepcopy(rbfm_orthnorm)
        rbfm_proj_merge = copy.deepcopy(rbfm_orthnorm[0])
        for i, rbfm_orthnorm_i in enumerate(rbfm_orthnorm):
            rbfm_orthnorm_i_ip = rbfm_orthnorm_i.inner_product(rbfm)
            for i2,_ in enumerate(rbfm_proj[i].radial_basis_function_list):
                m = rbfm_proj[i].radial_basis_function_list[i2].multiplier
                if rbfm_orthnorm_i_ip == 0:
                    rbfm_proj[i].radial_basis_function_list[i2].multiplier = 0
                elif ( (m != 0) ):
                    rbfm_proj[i].radial_basis_function_list[i2].multiplier = np.sign(m) * np.sign(rbfm_orthnorm_i_ip) * np.exp( np.log(np.abs(m)) + np.log(np.abs(rbfm_orthnorm_i_ip)) )
        # merge the multipliers and their corresponding orthogonalized and normalized radial basis functions
        for i2,_ in enumerate(rbfm_proj_merge.radial_basis_function_list):
            rbfm_proj_merge.radial_basis_function_list[i2].multiplier = np.sum([
                rbfm_proj[i].radial_basis_function_list[i2].multiplier
                for i,_ in enumerate(rbfm_proj)
            ])
        
        return rbfm_proj_merge

    def project(self, rbfm):
        '''
        project a radial basis function combination onto the subspace spanned by the current list of radial basis functions. this uses the projection matrix with gram matrix inversion.
        ---
        rbfm: RadicalBasisFunctionMixture object
        ---
        return: RadicalBasisFunctionMixture object
        '''
        ip_ary_new_this = np.array([
            rbfm.inner_product(
                RadicalBasisFunctionMixture( radial_basis_function_list= [r] )
            )
            for r in self.radial_basis_function_list
        ])[None,:]
        if self.gram_matrix_inv is None:
            gram_matrix = np.array([
                [r1.inner_product(r2) for r2 in self.radial_basis_function_list]
                for r1 in self.radial_basis_function_list
            ])
            self.gram_matrix_inv = np.linalg.inv(gram_matrix)
        proj_multipliers = ip_ary_new_this @ self.gram_matrix_inv
        rbfm_proj = RadicalBasisFunctionMixture( radial_basis_function_list= copy.deepcopy(self.radial_basis_function_list) )
        for i,_ in enumerate(rbfm_proj.radial_basis_function_list):
            m = rbfm_proj.radial_basis_function_list[i].multiplier
            if m != 0:
                if proj_multipliers[0,i] == 0:
                    rbfm_proj.radial_basis_function_list[i].multiplier = 0
                else:
                    rbfm_proj.radial_basis_function_list[i].multiplier = np.sign(m) * np.sign(proj_multipliers[0,i]) * np.exp( np.log(np.abs(m)) + np.log(np.abs(proj_multipliers[0,i])) )

        rbfm_proj_ip = np.squeeze( proj_multipliers @ ip_ary_new_this.T ).item()
        return rbfm_proj, rbfm_proj_ip


    def cosine_similar2subspace(
            self,
            rbfm,
        ):
        '''
        cosine similarity between a radial basis function mixture and the subspace spanned by the current list of radial basis functions
        ---
        rbfm: RadicalBasisFunctionMixture object
        '''
        rbfm_proj, rbfm_proj_ip = self.project(rbfm)
        orig_proj_ip = rbfm_proj.inner_product(rbfm)
        logging.debug(f'orig_proj_ip={orig_proj_ip}')
        logging.debug(f'rbfm_proj_ip={rbfm_proj_ip}')
        orig_ip = rbfm.inner_product(rbfm)
        logging.debug(f'orig_ip={orig_ip}')

        cosine_similarity = np.exp( np.log(orig_proj_ip) - 0.5 * np.log(orig_ip) - 0.5 * np.log(rbfm_proj_ip) )

        return cosine_similarity
