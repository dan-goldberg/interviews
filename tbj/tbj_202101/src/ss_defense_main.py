#!/usr/bin/env python3

"""
This python script runs the Multilevel Logistic Regression model with all preprocessing all self-contained 
within this single file. In reality this was run with a directory tree and different classes and functions 
defined in a sensible structure, instad of all in one file. See the actual repository at 
https://github.com/dan-goldberg/interviews/tree/master/tbj/tbj_202101
"""

# imports 

from uuid import uuid1
import datetime
import pickle
import json
import os
from pathlib import Path
import logging

import matplotlib.image as mpimg # https://matplotlib.org/3.3.3/tutorials/introductory/images.html
# https://image.online-convert.com/convert/svg-to-png
from PIL import Image

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.special import expit
import pystan
import xarray as xr

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn import pipeline #import make_pipeline, Pipeline

# Modelling
from utils.ml_wrappers import StanLogisticMultilevel1

# model evaluation 
from sklearn import metrics #log_loss, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn import model_selection # cross_val_score

# hyperparam optimization
import skopt

ASSETS_DIR = '../assets'
DATA_DIR = '../data' # make sure the raw data is in this directory as a CSV
MODEL_REGISTRY_DIR = '.'
MODEL_REGISTRY_FILE = f'{MODEL_REGISTRY_DIR}/model_registry.jsonl'

# Configure Logging

logger = logging.getLogger('OAA Model')
# create console handler with a higher log level
ch = logging.StreamHandler()
# create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(ch)


### GEOMETRIC CALCULATIONS FOR PREPROCESSING

class Geometry:
    
    @staticmethod
    def get_launch_trajectory_vector(horizonal_angle):
        """[summary]

        Args:
            horizonal_angle ([type]): [description]

        Returns:
            [type]: [description]
        """
        hypotenuse = 250
        y = hypotenuse * np.cos(Geometry.degrees_to_radians(horizonal_angle))
        x = hypotenuse * np.sin(Geometry.degrees_to_radians(horizonal_angle))
        return x, y

    @staticmethod
    def get_orthogonal_projection_vector(vector, projection_surface):
        # https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
        return np.dot((np.dot(vector, projection_surface) / np.dot(projection_surface, projection_surface)), projection_surface)

    @staticmethod
    def degrees_to_radians(degrees):
        return degrees * 2 * np.pi / 360

    @staticmethod
    def radians_to_degrees(radians):
        return (radians * 360) / (2 * np.pi)

    @staticmethod
    def likely_interception_point(
        launch_speed, 
        horizonal_angle, 
        player_x, 
        player_y,
        player_speed=21.0
        ):
        # need to find the vector paralell with trajectory vector 
        # which minimizes the ball flight time given the constraints
        # player movement time = ball flight time

        player_speed = player_speed # feet per second
        ball_speed = Geometry.mph_to_feet_per_second(launch_speed*0.95)
            # this should probably be a more complicated 
            # function of launch speed and how close to home the ball landed
        launch_trajectory = Geometry.get_launch_trajectory_vector(horizonal_angle)
        m = launch_trajectory[1]/launch_trajectory[0] # y/x to get m from y=mx

        # coefficients of quadratic equation ax^2 + bx + c = 0
        a = (1 - (player_speed / ball_speed)**2) * (m**2 + 1)
        b = -2 * (player_x + ( m * player_y))
        c = (player_x**2) + (player_y**2)

        # solve quadratic formula
        roots = np.roots([a, b, c]) 
        roots_is_real = ~np.iscomplex(roots)
        if not roots_is_real.any():
            return None # if there are no real roots, the player cannot intercept the ball given assumptions
        else:
            roots = roots[roots_is_real] # only look at the real root(s)
            preferred_root_index = np.argmin(roots**2) # take the root closer to home plate
            min_time_x = roots[preferred_root_index] 
            min_time_y = min_time_x*m
            return np.array([min_time_x, min_time_y])

    @staticmethod
    def get_angle_between_vectors(vector_1, vector_2):
        unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        angle = np.arccos(dot_product)
        return Geometry.radians_to_degrees(angle)

    @staticmethod
    def player_approach_angle(player_position, throw_position, base_position):
        return Geometry.get_angle_between_vectors(
            throw_position - player_position, 
            base_position - player_position
            )

    @staticmethod
    def distance_between_coords(vector_1, vector_2):
        return np.linalg.norm(vector_1 - vector_2)

    @staticmethod
    def mph_to_feet_per_second(mph):
        # feet_per_mile = 5280
        return (mph * 5280) / (60**2)
    
class Diamond:
    """
    This class holds facts about MLB ballpark dimensions for use in preprocessing and visualizing observations.
    """
    
    def __init__(self, 
                 figsize=(8,8), 
                 scatter_kwargs=dict(s=10, alpha=1.0)
                ):
        """
        A class that exposes a to-scale plot of a pro diamond and
        helper function to plot scatter points with units in feet from home plate.
        
        All input units are feet with the plane centred on home plate. 
        
        You can scatter plot coordinates and also line segments.
        """
        # base image details
        self.img = Image.open(f'{ASSETS_DIR}/BaseballDiamondScale.jpg')    # Open image as PIL image object
        self.HOME_X = 647
        self.HOME_Y = 1061
        self.SECOND_BASE_Y = 767
        
        # constants for plotting coordinates
        FEET_BETWEEN_HOME_AND_SECOND = np.math.pow(2 * np.math.pow(90,2), 1/2) 
        self.UNITS_PER_FEET = (self.HOME_Y - self.SECOND_BASE_Y) / FEET_BETWEEN_HOME_AND_SECOND # 127.279 feet
        
        # coordinates with attributes
        self.scatter_kwargs = scatter_kwargs
        self.coords = [] #empty list of coordinates
        
        # for line segment
        self.lines_x = []
        self.lines_y = []

        # base positions in feet

        self.HOME_VECTOR = np.array([0, 0])

        self.FIRST_BASE_VECTOR = self._calc_vector_in_feet(
            *np.array([
                self.HOME_X + (self.HOME_Y-self.SECOND_BASE_Y)/2,
                self.HOME_Y - (self.HOME_Y-self.SECOND_BASE_Y)/2
            ])
        )

        self.SECOND_BASE_VECTOR = (0, FEET_BETWEEN_HOME_AND_SECOND)

        self.THIRD_BASE_VECTOR = self._calc_vector_in_feet(
            *np.array([
                self.HOME_X - (self.HOME_Y-self.SECOND_BASE_Y)/2,
                self.HOME_Y - (self.HOME_Y-self.SECOND_BASE_Y)/2
            ])
        )
        
    def set_coord(self, x, y, **kwargs):
        self.coords.append((self._calc_coord(x,y), dict(**kwargs)))
        
    def _calc_coord(self, x, y):
        # x and y in feet from home
        return (x*self.UNITS_PER_FEET)+self.HOME_X, ((-y*self.UNITS_PER_FEET)+self.HOME_Y)
        
    def _calc_vector_in_feet(self, x, y):
        # x and y in coords
        return (x - self.HOME_X) / self.UNITS_PER_FEET, (y - self.HOME_Y) / (-self.UNITS_PER_FEET)
    
    def plot_line_segment(self, coord1, coord2):
        reshaped = list(zip(self._calc_coord(*coord1), self._calc_coord(*coord2)))
        self.lines_x.append(list(reshaped[0]))
        self.lines_y.append(list(reshaped[1]))
        
    def show(self, title=None):
        fig = plt.figure(figsize=(9,9))
        plt.axis('off')
        plt.imshow(self.img)
        coords = np.array(self.coords)
        for line_x, line_y in zip(self.lines_x, self.lines_y):
            plt.plot(line_x, line_y)
        if self.coords:
            for coord in self.coords:
                plt.scatter(coord[0][0], coord[0][1], **dict(**coord[1], **self.scatter_kwargs))
        if title:
            plt.title(title)
        plt.legend()
        fig.show()

### Preprocessing functionality, from loading to transforming, to breaking up into component inputs/outputs.

class Preprocessing:
    """
    Methods in this class load the raw data, append new columns, transform columns, and break up the dataset in X, y, and index.
    """

    PLAYER_SPEED = 21.0 
    diamond = Diamond()
       
    @staticmethod      
    def shortstop_global_preprocessing(df):
        """
        Takes in the raw data as a pandas DataFrame, and outputs it preprocessed.
        """
        
        # remove bad data
        df = df.dropna(subset=['fieldingplay']) # there are 2 rows with NULL in fieldingplay column

        # reformat columns
        df.loc[:, 'fieldingplay'] = df['fieldingplay'].astype(int).astype(str)
        df.loc[:, 'fielded_pos'] = df['fielded_pos'].astype(int).astype(str)

        # boolean flags for conveneince
        df.loc[:, 'team_made_out'] = df['playtype'].apply(lambda x : x == 'hit_into_play')
        df.loc[:, 'ss_in_play'] = df['fieldingplay'].apply(lambda x: '6' in x)
        df.loc[:, 'ss_made_play'] = df['fielded_pos'].apply(lambda x: x == '6')
        df.loc[:, 'ss_made_out'] = df.apply(lambda x: (x['ss_made_play'] & x['team_made_out']), axis=1)
        df.loc[:, 'non_ss_infield_field_error'] = df.apply(lambda x: (x['eventtype'] == 'field_error') and (x['fielded_pos'] in ['1','2','3','4','5']), axis=1)
        
        # these are the plays we'll want to give/take credit to/from the shortstop
        df.loc[:, 'ss_evaluation_play'] = df.apply(lambda x: (not (x['team_made_out'] or x['non_ss_infield_field_error'])) or (x['ss_made_play']), axis=1)
        
        # derive valuable vectors for downstream use
        df.loc[:, 'player_point'] = df.apply(lambda x: x[['player_x','player_y']].values, axis=1)
        df.loc[:, 'launch_trajectory_vector'] = df['launch_horiz_ang'].apply(Geometry.get_launch_trajectory_vector)
        df.loc[:, 'landing_location_point'] = df.apply(lambda x: x[['landing_location_x','landing_location_y']].values, axis=1)

        # what is the shortest path for the player to the trajectory line of the ball
        df.loc[:, 'orthogonal_interception_point'] = df.apply(lambda x: Geometry.get_orthogonal_projection_vector(
            x['player_point'], 
            x['launch_trajectory_vector']
        ), axis=1)

        # if the player has extra time he can charge the ball, so let's get the likely interception point if that's the case
        # this finds the point where the player's time to the spot is the same as the ball's time to the spot
        df.loc[:, 'inferred_interception_point'] = df.apply(lambda x: Geometry.likely_interception_point(
            launch_speed = x['launch_speed'],
            horizonal_angle = x['launch_horiz_ang'],
            player_x = x['player_x'],
            player_y = x['player_y'],
            player_speed = Preprocessing.PLAYER_SPEED
        ) , axis=1)

        # given the player speed assumption, many balls will not be interceptable; in these cases take the orthogonal vector point on the ball trajectory line
        df.loc[:, 'utilized_interception_point'] = df.apply(lambda x: x['inferred_interception_point'] if x['inferred_interception_point'] is not None else x['orthogonal_interception_point'], axis=1)

        # using speed assumptions, derive time-to-interception for utilized point
        df.loc[:, 'player_time_to_point_of_interest'] = df.apply(lambda x: Geometry.distance_between_coords(x['utilized_interception_point'], x['player_point'])/Preprocessing.PLAYER_SPEED, axis=1)
        df.loc[:, 'ball_time_to_point_of_interest'] = df.apply(lambda x:  np.linalg.norm(x['utilized_interception_point'])/(Geometry.mph_to_feet_per_second(x['launch_speed']*0.95)), axis=1)
        df.loc[:, 'player_time_minus_ball_time'] = df['player_time_to_point_of_interest'] - df['ball_time_to_point_of_interest']
        df.loc[:, 'distance_from_point_of_interest_to_landing_location'] = df.apply(lambda x: Geometry.distance_between_coords(x['utilized_interception_point'], x['landing_location_point']), axis=1)

        # once the player has the ball he still has to get it to the base.
        # let's try to guess which base he will want to throw it to
        bases_map = {
            '1':Preprocessing.diamond.FIRST_BASE_VECTOR,
            '2':Preprocessing.diamond.SECOND_BASE_VECTOR,
            '3':Preprocessing.diamond.THIRD_BASE_VECTOR,
            '4':Preprocessing.diamond.HOME_VECTOR
        }

        def base_of_interest_switch(df, point):
            candidate_bases = ['1']
            if (df['runner_on_first']) & (not df['is_runnersgoing']):
                candidate_bases.append('2')
            if (df['runner_on_first']) & (df['runner_on_second']) & (not df['is_runnersgoing']):
                candidate_bases.append('3')
            if (df['runner_on_first']) & (df['runner_on_second']) & (df['runner_on_third']) & (not df['is_runnersgoing']):
                candidate_bases.append('4')
            distances = [Geometry.distance_between_coords(point, bases_map[base]) for base in candidate_bases]
            adjusted_distances = []
            for i, distance in enumerate(distances): 
                # adjust distances for non-1B up since runners get leadoffs
                # essentially this is the same as saying the SS has less time
                # to get the ball to the base if it's not 1B
                if candidate_bases[i] == '1':
                    coef = 1.0
                else:
                    coef = 1.10 # like saying the runner gets to the base 10% faster if it's not 1B
                adjusted_distances.append(distance * coef)
            return candidate_bases[np.argmin(adjusted_distances)]


        df.loc[:, 'base_of_interest'] = df.apply(lambda x: base_of_interest_switch(x, x['utilized_interception_point']), axis=1)
        df.loc[:, 'base_of_interest_point'] = df.apply(lambda x: bases_map[x['base_of_interest']], axis=1)
        
        # based on the base of interest, how far is the throw
        df.loc[:, 'player_distance_from_interception_point_to_base_of_interest'] = df.apply(lambda x: Geometry.distance_between_coords(
            x['utilized_interception_point'],
            x['base_of_interest_point']
        ), axis=1)
        
        # based on base of interest, let's think about whether his momentum will be taking him towards or away from the base
        df.loc[:, 'player_angle_from_interception_point_to_base_of_interest'] = df.apply(lambda x: Geometry.player_approach_angle(
            player_position = x['player_point'], 
            throw_position = x['utilized_interception_point'], 
            base_position = x['base_of_interest_point']
        ), axis=1)
        
        return df  


    @staticmethod     
    def shortstop_prep_inputs(df, feature_columns):
        """
        Takes in the preprocessed full dataframe, and breaks it up into X, y, and index components.
        """
        
        df = df[df['ss_evaluation_play']] # only take plays we care about

        df = df.dropna(subset=feature_columns)
        df = df.set_index('id')
        
        target_name = 'ss_made_out'
        
        X, y = df[feature_columns], df[target_name].astype(int)
        return df, X, y, feature_columns, target_name, df.index

    @staticmethod      
    def experiment_prep():
        """
        Define the columns and runs the load+preprocessing to get all assets needed to run experiment/model.
        """
        
        # Global Preprocessing to generate global dataset
        df = pd.read_csv(f'{DATA_DIR}/shortstopdefense.csv')
        df = Preprocessing.shortstop_global_preprocessing(df)

        feature_columns = [
            'launch_speed',
            'launch_vert_ang',
            'landing_location_radius', # maybe
            'hang_time', # maybe
            'player_time_to_point_of_interest',
            'ball_time_to_point_of_interest', # maybe this should be eliminated b/c is superfluous w/ player_time_to_point_of_interest and player_time_minus_ball_time
            'player_time_minus_ball_time', # always positive, lots of 0s for plays made
            'distance_from_point_of_interest_to_landing_location', # hopefully can distinguish short hops from good hops
            'player_angle_from_interception_point_to_base_of_interest',
            'player_distance_from_interception_point_to_base_of_interest'
        ]
        log_scale_transform_feature_columns = [
            'hang_time',
            'distance_from_point_of_interest_to_landing_location'
        ]
        standard_scalar_feature_columns = [
            'launch_speed',
            'launch_vert_ang',
            'landing_location_radius',
            'player_time_to_point_of_interest',
            'ball_time_to_point_of_interest',
            'player_time_minus_ball_time',
            'player_angle_from_interception_point_to_base_of_interest',
            'player_distance_from_interception_point_to_base_of_interest'
        ]

        df, X, y, feature_columns, target_name, index = Preprocessing.shortstop_prep_inputs(df, feature_columns)

        # Define params that are consistent between experiments

        param_payload = dict(
            X = X,
            y = y,
            target_name = target_name,
            feature_columns = feature_columns,
            feature_preprocessing = ColumnTransformer([
                    ('log_scale_transform', Preprocessing.LogScaleTransformer(), log_scale_transform_feature_columns),
                    ('standard_scalar', StandardScaler(), standard_scalar_feature_columns)
                ], remainder='passthrough'),
            objective = metrics.make_scorer(
                metrics.log_loss,
                needs_proba=True,
                greater_is_better=False
                )
        )

        return df, X, y, feature_columns, target_name, index, param_payload

    @staticmethod
    def LogScaleTransformer():
        return pipeline.make_pipeline(
            FunctionTransformer(np.log, np.exp),
            StandardScaler()
        )      

### Classes for training the model

class StanLogisticMultilevel1:
    # implement the model itself (without the variable effects)

    def __init__(self, model_code=None, model_file=None, fixed_effects_columns=[], **kwargs):
        if model_code and model_file:
            raise Exception('You input both a model code and a model file. Which is it? Pick one.')
        elif model_code:
            self.model = pystan.StanModel(model_code=model_code, **kwargs)
        elif model_file:
            self.model = pystan.StanModel(file=model_file, **kwargs)
        else:
            raise Exception('Must input either model_code or model_file.')

        self.fixed_effects_columns = fixed_effects_columns
        self.named_steps = {
            'model': self.model,
            'feature_preprocessing': None
        }

    def fit(self, data, iter=1000, chains=4, **kwargs):
        self.fit_results = self.model.sampling(data=data, iter=iter, chains=chains, **kwargs)
        self.params = self.fit_results.extract(permuted=True)
        #print(self.fit.stansummary())

    def predict(self, X):
        return self._predict_with_samples(X).mean(0) > 0.5

    def predict_proba(self, X):
        return self._predict_with_samples(X).mean(0)

    def _predict_with_samples(self, X):
        self.params = self.fit_results.extract(permuted=True)  # return a dictionary of arrays
        mu = self.params['fixed_effects']
        return expit(np.matmul(mu, X.T))

    def get_params(self):
        return self.params  

class Training:
    
    @staticmethod
    def train_model(df, X, y, feature_columns, target_name, index, param_payload):

        stan_model_code = """
            // This model keeps every feature and parameter separate. I tried doing it in a more vectorized way
            // but ran into serious convergence issues. So the model is more bespoke (classic Bayesian) but
            // actually works on my data, which is nice!
            data {
                int<lower=0> num_samples;
                int<lower=1> num_features;

                vector[num_samples] feature1;
                vector[num_samples] feature2;
                vector[num_samples] feature3;
                vector[num_samples] feature4;
                vector[num_samples] feature5;
                vector[num_samples] feature6;
                vector[num_samples] feature7;
                vector[num_samples] feature8;
                vector[num_samples] feature9;
                vector[num_samples] feature10;

                int<lower=0,upper=1> labels[num_samples];

                // variable effects
                int<lower=1> num_levels;
                int<lower=1,upper=num_levels> level[num_samples];
            }
            parameters {
                real slope1;
                real slope2;
                real slope3;
                real slope4;
                real slope5;
                real slope6;
                real slope7;
                real slope8;
                real slope9;
                real slope10;
                real bias;
                
                // variable effects
                real<lower=0> sigma;
                real shortstop_effect[num_levels];
            }
            model {
                vector[num_samples] x_beta_ll;

                slope1 ~ normal(0, 1);
                slope2 ~ normal(0, 1);
                slope3 ~ normal(0, 1);
                slope4 ~ normal(0, 1);
                slope5 ~ normal(0, 1);
                slope6 ~ normal(0, 1);
                slope7 ~ normal(0, 1);
                slope8 ~ normal(0, 1);
                slope9 ~ normal(0, 1);
                slope10 ~ normal(0, 1);
                bias ~ normal(0, 1);

                sigma ~ exponential(1);
                for (l in 1:num_levels) {
                    shortstop_effect[l] ~ normal(0, sigma);
                }
                for (n in 1:num_samples) {
                    x_beta_ll[n] = bias
                        + feature1[n] * slope1 
                        + feature2[n] * slope2 
                        + feature3[n] * slope3 
                        + feature4[n] * slope4 
                        + feature5[n] * slope5 
                        + feature6[n] * slope6 
                        + feature7[n] * slope7 
                        + feature8[n] * slope8 
                        + feature9[n] * slope9 
                        + feature10[n] * slope10 
                        + shortstop_effect[level[n]];
                }
                labels ~ bernoulli_logit(x_beta_ll);
            }
        """

        model = StanLogisticMultilevel1(model_code=stan_model_code, fixed_effects_columns=feature_columns)

        experiment_details = dict(
            model = model,
            space = None,
            experiment_name = "Multilevel Logistic Regression v1",
            experiment_description = f"""
                Hierarchical model same as vanilla logistic regression plus a random intercept 
                for the shortstop identity, trained with NUTS MCMC sampler in Stan. The hope
                is to control for confounder of player identity and ground ball profile (difficulty),
                i.e. the pitchers on the player's team give up hard ground balls more often. 

                Input Features: {feature_columns}
                Ouptut: {target_name}
                """,
            **param_payload
        )

        # reindex player id
        oe = OrdinalEncoder(dtype=int)
        df.loc[:, 'playerid_cat'] = oe.fit_transform(df[['playerid']]) 
        levels = df['playerid_cat'].values + 1 # reindex with 1-index
        num_levels = np.unique(levels).shape[0]
        
        # inner-loop preprocessing pipeline (log-standarize or standardize)
        column_transformer = param_payload['feature_preprocessing']
        X_t = column_transformer.fit_transform(X[feature_columns])

        # bespoke data dictionary for the bespoke Stan model
        data = {
            'num_samples': X_t.shape[0],
            'num_features': X_t.shape[1],
            'feature1': X_t[:,0],
            'feature2': X_t[:,1],
            'feature3': X_t[:,2],
            'feature4': X_t[:,3],
            'feature5': X_t[:,4],
            'feature6': X_t[:,5],
            'feature7': X_t[:,6],
            'feature8': X_t[:,7],
            'feature9': X_t[:,8],
            'feature10': X_t[:,9],
            'num_levels': num_levels,
            'level': levels,
            'labels':y
        }

        # fit model
        model.fit(data, control={'max_treedepth': 15})

        # evaluate model
        params = model.params  # return a dictionary of arrays
        bias = params['bias']
        slope1 = params['slope1']
        slope2 = params['slope2']
        slope3 = params['slope3']
        slope4 = params['slope4']
        slope5 = params['slope5']
        slope6 = params['slope6']
        slope7 = params['slope7']
        slope8 = params['slope8']
        slope9 = params['slope9']
        slope10 = params['slope10']
        level_param = params['shortstop_effect']

        # get predictions
        preds_proba = expit(
            bias.reshape(-1,1) \
            + slope1.reshape(-1,1)*X_t[:,0].reshape(1,-1) \
            + slope2.reshape(-1,1)*X_t[:,1].reshape(1,-1) \
            + slope3.reshape(-1,1)*X_t[:,2].reshape(1,-1) \
            + slope4.reshape(-1,1)*X_t[:,3].reshape(1,-1) \
            + slope5.reshape(-1,1)*X_t[:,4].reshape(1,-1) \
            + slope6.reshape(-1,1)*X_t[:,5].reshape(1,-1) \
            + slope7.reshape(-1,1)*X_t[:,6].reshape(1,-1) \
            + slope8.reshape(-1,1)*X_t[:,7].reshape(1,-1) \
            + slope9.reshape(-1,1)*X_t[:,8].reshape(1,-1) \
            + slope10.reshape(-1,1)*X_t[:,9].reshape(1,-1) \
            + np.stack([level_param[:, l-1] for l in levels]).T
        )

        preds = (preds_proba > 0.5).astype(int)

        preds_proba_mean = preds_proba.mean(axis=0)
        preds_mean = (preds_proba_mean > 0.5).astype(int)

        result_logloss = metrics.log_loss(y, preds_proba_mean)
        result_accuracy = metrics.accuracy_score(y, preds_mean)

        print(f'GLMM Logloss: {result_logloss}')
        print(f'GLMM Accuracy: {result_accuracy}')

        # convert preds_proba into sklearn style 
        sk_preds_proba_mean = np.concatenate([ # Stan model only outputs prob of class 1
            1-preds_proba_mean.reshape(-1,1), 
            preds_proba_mean.reshape(-1,1)
        ], axis=1)

        model_details = {
            'model':model,
            'model_name': 'Multilevel Logistic Regression',
            'id': str(uuid1()),
            'df': df,
            'X_t': X_t,
            'y': y,
            'preds_proba_mean': preds_proba_mean,
            'preds_proba': preds_proba,
            'preds_mean': preds_mean,
            'preds': preds
        }

        model_details = Training.summarize_model(model_details, y, preds_mean, sk_preds_proba_mean)

        return model_details

    @staticmethod
    def summarize_model(model_details, y, preds, preds_proba, bins=100):
        # calculate brier score
        brier_score = brier_score_loss(y.values, preds_proba[:,1])
        # save to model_details payload
        model_details['brier'] = brier_score
        # display summary
        print(f'{model_details["model_name"]}: {brier_score: .3f} Brier Score\n')
        Training.plot_probability_calibration_curve(y, preds, preds_proba, model_details['model_name'], bins=bins)
        return model_details
        
    @staticmethod
    def plot_probability_calibration_curve(y, preds, preds_prob, name, bins=10):
        
        logloss_score = log_loss(y, preds_prob)
        
        print("\tPrecision: %1.3f" % precision_score(y, preds))
        print("\tRecall: %1.3f" % recall_score(y, preds))
        print("\tF1: %1.3f" % f1_score(y, preds))
        print("\tLog-Loss: %1.3f" % logloss_score)
        print("\tAccuracy: %1.3f\n" % accuracy_score(y, preds))

        prob_pos = preds_prob[:, 1]

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y, prob_pos, n_bins=10)
        
        fig = plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                label="%s (%1.3f)" % (name, logloss_score))

        ax2.hist(prob_pos, range=(0, 1), bins=bins, label=name,
                histtype="step", lw=2)
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()

    @staticmethod
    def summarize_outs_above_average(df, y, preds_proba, min_prob_for_opportunity = 0.005):
        """
        This function prepares a dataframe that summarizes the OAA metric, given a model's probability predictions.
        """

        model_output = xr.Dataset({
            'ss_out_probability':(['samples','observations'], preds_proba),
            'OAA': (['samples','observations'], y.values-preds_proba),
            'opportunities': (['samples', 'observations'], (preds_proba > min_prob_for_opportunity).astype(int)), # over a 0.5% chance
            'observed_out': ('observations', y),
            'playerid': ('observations', df['playerid'].values)
        })
        
        return model_output

    @staticmethod
    def generate_player_level_metrics(oaa):
        # aggregate across observations to get player-level stats
        player_summary = oaa.groupby('playerid')\
                            .sum(dim='observations')

        # aggregate across samples to get bayesian flavour of metrics
        player_summary['OAA_mean'] = player_summary.OAA.mean(dim='samples')
        player_summary['OAA_std'] = player_summary.OAA.std(dim='samples')
        player_summary['opportunities'] = player_summary.opportunities.mean(dim='samples')

        # convert to pandas dataframe
        player_summary = player_summary[['opportunities','OAA_mean','OAA_std']].to_dataframe().sort_values('OAA_mean', ascending=False)
        player_summary['OAA_per_Opp'] = player_summary['OAA_mean'] / player_summary['opportunities']

        # prep leaderboard with rank
        player_summary = player_summary.reset_index().reset_index().rename(columns={'index':'rank'})
        player_summary.loc[:, 'rank'] = player_summary['rank'] + 1
        player_summary = player_summary.set_index('rank')

        # format output
        format_1d = "{0:.1f}".format
        format_2d = "{0:.2f}".format
        format_3d =  "{0:.3f}".format
        player_summary[['opportunities']] = player_summary[['opportunities']].applymap(format_1d)
        player_summary[['OAA_mean','OAA_std']] = player_summary[['OAA_mean','OAA_std']].applymap(format_2d)
        player_summary[['OAA_per_Opp']] = player_summary[['OAA_per_Opp']].applymap(format_3d)

        # save file
        player_summary.to_csv('ss_OAA.csv')
        # print top 20
        print(player_summary[:20].to_markdown())
    

def main():
    logger.info('Begin!')
    logger.info('Starting: Preprocessing')
    df, X, y, feature_columns, target_name, index, param_payload = Preprocessing.experiment_prep()
    logger.info('Finished: Preprocessing')
    logger.info('Starting: Training Model')
    model_details = Training.train_model(df, X, y, feature_columns, target_name, index, param_payload)
    logger.info('Finished: Training Model')
    logger.info('Starting: Calculating OAA')
    oaa = Training.summarize_outs_above_average(model_details['df'], model_details['y'], model_details['preds_proba'])
    Training.generate_player_level_metrics(oaa)
    logger.info('Finished!')

if __name__ == '__main__':
    main()