import torch
import torch.nn as nn
import src.dvrl.aesmc.random_variable as rv
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, obs_dim, H, out_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(obs_dim, H)
        self.linear2 = nn.Linear(H, out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(nn.Module):
    def __init__(self, input_dim, H, obs_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, H)
        self.linear2 = nn.Linear(H, obs_dim)

    def forward(self, x):
        # I removed the relu layer at the final output layer
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class VRNN_transition(nn.Module):  # P(z|h,a)
    def __init__(self, h_dim, z_dim, action_encode_dim):
        super().__init__()
        self.prior = nn.Sequential(
            nn.Linear(h_dim + action_encode_dim, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.action_encode_dim = action_encode_dim
        self.h_dim = h_dim

    def forward(self, previousLatentState, obsActEncode):
        """Outputs the prior probability of z_t.
        Inputs:
            - previousLatentState containing at least
                `h`     [batch, particles, h_dim]
        """

        batch_size, num_particles, _ = previousLatentState.h.size()

        input = torch.cat([
            previousLatentState.h,
            obsActEncode.encoded_action], 2).view(-1, self.h_dim + self.action_encode_dim)

        prior_t = self.prior(input)
        prior_mean_t = self.prior_mean(prior_t).view(batch_size, num_particles, -1)
        prior_std_t = self.prior_std(prior_t).view(batch_size, num_particles, -1)

        prior_dist = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(
                mean=prior_mean_t,
                variance=prior_std_t
            ))

        return prior_dist


class VRNN_deterministic_transition(nn.Module):  # get the next h with RNN
    def __init__(self, z_dim, observation_encode_dim, h_dim, action_encode_dim):
        super().__init__()
        # do an extra encode for Z before feeding into the RNN
        self.encodeZ = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())

        self.rnn = nn.GRUCell(h_dim + observation_encode_dim + action_encode_dim, h_dim)

        self.observation_encode_dim = observation_encode_dim
        self.action_encode_dim = action_encode_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

    def forward(self, previousLatentState, latentState, obsActEncode):
        # get batch size and # of particles
        batch_size, num_particles, _ = latentState.z.size()

        encoded_observation = obsActEncode.encoded_observation
        encoded_Z = self.encodeZ(latentState.z.view(-1, self.z_dim)).view(batch_size, num_particles, self.h_dim)
        encoded_action = obsActEncode.encoded_action

        # encoded_Z has shape [1,15,256]
        input = torch.cat([
            encoded_observation,
            encoded_Z,
            encoded_action], 2).view(-1, self.h_dim + self.observation_encode_dim + self.action_encode_dim)

        h = self.rnn(input, previousLatentState.h.view(-1, self.h_dim))
        # add encoded Z and new h into latent state
        latentState.encoded_Z = encoded_Z.view(batch_size, num_particles, -1)
        latentState.h = h.view(batch_size, num_particles, self.h_dim)

        return latentState


class VRNN_emission(nn.Module):  # decoding
    def __init__(self, h_dim, action_encode_dim, hidden_dim, observation_dim, observation_encode_dim):
        super().__init__()

        encoding_dimension = h_dim + h_dim + action_encode_dim

        # To do
        self.dec_mean = nn.Sigmoid()
        self.dec_std = None
        self.dec = Decoder(observation_encode_dim, hidden_dim, observation_dim)

        self.linear_obs_decoder = nn.Sequential(
            nn.Linear(encoding_dimension, observation_encode_dim),
            nn.ReLU())

        self.observation_encode_dim = observation_encode_dim
        self.observation_dim = observation_dim
        self.action_encode_dim = action_encode_dim
        self.h_dim = h_dim

    def forward(self, previousLatentState, latentState, obsActEncode):
        """
        Returns: emission_dist [batch-size, num_particles, n_observations]
        """
        batch_size, num_particles, encoded_Z_dim = latentState.encoded_Z.size()

        # first change z, h, a to observation_encode_dim
        # then change to n_observation
        # needs two steps -- similar to encode function (first encode then turn to z)
        dec_t = self.linear_obs_decoder(torch.cat([
            latentState.encoded_Z,
            previousLatentState.h,
            obsActEncode.encoded_action
        ], 2).view(-1, encoded_Z_dim + self.h_dim + self.action_encode_dim))

        dec_t = self.dec(dec_t.view(-1, self.observation_encode_dim))
        dec_mean_t = self.dec_mean(dec_t)
        dec_mean_t = dec_mean_t.view(batch_size, num_particles, self.observation_dim)

        # condition
        emission_dist = rv.StateRandomVariable(
            observation=rv.MultivariateIndependentPseudobernoulli(
                probability=dec_mean_t))

        return emission_dist


class VRNN_proposal(nn.Module):  # from encoded observation to z's distribution
    #
    # z_dim: latent state dimension (=256 in original paper)
    # h_dim: dimension of h, also 256
    def __init__(self, z_dim, h_dim, observation_encode_dim, action_encode_dim):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Linear(h_dim + observation_encode_dim + action_encode_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())
        self.observation_encode_dim = observation_encode_dim
        self.action_encode_dim = action_encode_dim
        self.h_dim = h_dim

    def forward(self, previousLatentState, obsActEncode):
        encoded_observation = obsActEncode.encoded_observation
        encoded_action = obsActEncode.encoded_action
        batch_size, num_particles, _ = encoded_observation.size()

        input = torch.cat([
            encoded_observation,
            previousLatentState.h,
            encoded_action
        ], 2).view(-1, self.observation_encode_dim + self.h_dim + self.action_encode_dim)

        # input shape = [num_particles, action_dim+obs_dim+h_dim]
        enc_t = self.enc(input)
        enc_mean_t = self.enc_mean(enc_t).view(batch_size, num_particles, -1)
        enc_std_t = self.enc_std(enc_t).view(batch_size, num_particles, -1)
        # enc_mean_t shape = [1,num_particles, h_dim] [1,15,256]

        proposed_state = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(
                mean=enc_mean_t,
                variance=enc_std_t
            ))
        return proposed_state


class VRNN_encoding(nn.Module):  # from sequence observation to lower-dimension encoded # encode Ot, at-1

    def __init__(self, n_observation, hidden_dim, observation_encode_dim, n_actions, action_encode_dim):
        super().__init__()

        self.observation_encoder = Encoder(n_observation, hidden_dim, observation_encode_dim)

        self.action_encoder = nn.Sequential(
            nn.Linear(n_actions, action_encode_dim),
            nn.ReLU())

        self.n_observation = n_observation
        self.n_actions = n_actions

    def forward(self, obsActEncode):
        # input of observation has dimension [1,observation_dim]

        encoded_observation = self.observation_encoder(obsActEncode.observation.view(-1, self.n_observation)).view(1, 1, -1)
        encoded_action = self.action_encoder(obsActEncode.action.view(-1, self.n_actions)).view(1, 1, -1)

        # output [1, 1, encoded_obdim] [1,1,encoded_actdim]，original paper = 1568 encoded_obs_dim (ok to change)
        obsActEncode.encoded_observation = encoded_observation
        obsActEncode.encoded_action = encoded_action

        return obsActEncode
