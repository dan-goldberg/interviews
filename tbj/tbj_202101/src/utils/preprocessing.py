import pandas as pd
import numpy as np

from .viz_utils import Diamond
from . import geometry

# sklearn preprocessing
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import pipeline #import make_pipeline, Pipeline
    
# This is a gross oversimplification of an infielder's speed to a ball,
# but will do for now. 21 ft/second was guessed based on the notes on 
# the Sprint Speed metric on BaseballSavant found here
# https://baseballsavant.mlb.com/sprint_speed_leaderboard
PLAYER_SPEED = 21.0 

diamond = Diamond()

def shortstop_global_preprocessing(df):
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
    df.loc[:, 'launch_trajectory_vector'] = df['launch_horiz_ang'].apply(geometry.get_launch_trajectory_vector)
    df.loc[:, 'landing_location_point'] = df.apply(lambda x: x[['landing_location_x','landing_location_y']].values, axis=1)

    # what is the shortest path for the player to the trajectory line of the ball
    df.loc[:, 'orthogonal_interception_point'] = df.apply(lambda x: geometry.get_orthogonal_projection_vector(
        x['player_point'], 
        x['launch_trajectory_vector']
    ), axis=1)

    # if the player has extra time he can charge the ball, so let's get the likely interception point if that's the case
    # this finds the point where the player's time to the spot is the same as the ball's time to the spot
    df.loc[:, 'inferred_interception_point'] = df.apply(lambda x: geometry.likely_interception_point(
        launch_speed = x['launch_speed'],
        horizonal_angle = x['launch_horiz_ang'],
        player_x = x['player_x'],
        player_y = x['player_y'],
        player_speed = PLAYER_SPEED
    ) , axis=1)

    # given the player speed assumption, many balls will not be interceptable; in these cases take the orthogonal vector point on the ball trajectory line
    df.loc[:, 'utilized_interception_point'] = df.apply(lambda x: x['inferred_interception_point'] if x['inferred_interception_point'] is not None else x['orthogonal_interception_point'], axis=1)

    # using speed assumptions, derive time-to-interception for utilized point
    df.loc[:, 'player_time_to_point_of_interest'] = df.apply(lambda x: geometry.distance_between_coords(x['utilized_interception_point'], x['player_point'])/PLAYER_SPEED, axis=1)
    df.loc[:, 'ball_time_to_point_of_interest'] = df.apply(lambda x:  np.linalg.norm(x['utilized_interception_point'])/(geometry.mph_to_feet_per_second(x['launch_speed']*0.95)), axis=1)
    df.loc[:, 'player_time_minus_ball_time'] = df['player_time_to_point_of_interest'] - df['ball_time_to_point_of_interest']
    df.loc[:, 'distance_from_point_of_interest_to_landing_location'] = df.apply(lambda x: geometry.distance_between_coords(x['utilized_interception_point'], x['landing_location_point']), axis=1)

    # once the player has the ball he still has to get it to the base.
    # let's try to guess which base he will want to throw it to
    bases_map = {
        '1':diamond.FIRST_BASE_VECTOR,
        '2':diamond.SECOND_BASE_VECTOR,
        '3':diamond.THIRD_BASE_VECTOR,
        '4':diamond.HOME_VECTOR
    }

    def base_of_interest_switch(df, point):
        candidate_bases = ['1']
        if (df['runner_on_first']) & (not df['is_runnersgoing']):
            candidate_bases.append('2')
        if (df['runner_on_first']) & (df['runner_on_second']) & (not df['is_runnersgoing']):
            candidate_bases.append('3')
        if (df['runner_on_first']) & (df['runner_on_second']) & (df['runner_on_third']) & (not df['is_runnersgoing']):
            candidate_bases.append('4')
        distances = [geometry.distance_between_coords(point, bases_map[base]) for base in candidate_bases]
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
    df.loc[:, 'player_distance_from_interception_point_to_base_of_interest'] = df.apply(lambda x: geometry.distance_between_coords(
        x['utilized_interception_point'],
        x['base_of_interest_point']
    ), axis=1)
    
    # based on base of interest, let's think about whether his momentum will be taking him towards or away from the base
    df.loc[:, 'player_angle_from_interception_point_to_base_of_interest'] = df.apply(lambda x: geometry.player_approach_angle(
        player_position = x['player_point'], 
        throw_position = x['utilized_interception_point'], 
        base_position = x['base_of_interest_point']
    ), axis=1)
    
    return df

def shortstop_prep_inputs(df, feature_columns=[
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
]):
    
    df = df[df['ss_evaluation_play']] # only take plays we care about
    df = df.set_index('id')
    
    target_name = 'ss_made_out'
    
    X, y = df[feature_columns], df[target_name].astype(int)
    return df, X, y, feature_columns, target_name, df.index

if __name__ == '__main__':
    df = pd.read_csv('../data/shortstopdefense.csv')
    df = shortstop_global_preprocessing(df)
    df, X, y, feature_columns, target_name, index = shortstop_prep_inputs(df)