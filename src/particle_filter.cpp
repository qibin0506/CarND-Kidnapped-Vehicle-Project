/**
 * particle_filter.cpp
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  
  std::default_random_engine gen;
  
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];
  
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);
  
  for (int i = 0; i< num_particles; i++) {
    Particle particle;
    
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    
  	particles.push_back(particle);
    weights.push_back(particle.weight);
  }
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  
  for (int i = 0; i < num_particles; i++) {
  	double x = particles[i].x;
    double y = particles[i].y;
    double theta = particles[i].theta;
    
    double prediction_x, prediction_y, prediction_theta;
    if (fabs(yaw_rate) < 0.0001) {
      prediction_x = x + velocity * cos(theta) * delta_t;
      prediction_y = y + velocity * sin(theta) * delta_t;
      prediction_theta = theta;
    } else {
      prediction_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      prediction_y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      prediction_theta = theta + yaw_rate * delta_t;
    }
    
    normal_distribution<double> dist_x(prediction_x, std_pos[0]);
    normal_distribution<double> dist_y(prediction_y, std_pos[1]);
    normal_distribution<double> dist_theta(prediction_theta, std_pos[2]);
    
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for (int i = 0; i < observations.size(); i++) {
    long cur_obs_x = observations[i].x;
    long cur_obs_y = observations[i].y;
    
    long min_distance = -1;
    int cur_id;
    
    for (int j = 0; j < predicted.size(); j++) {
      long cur_pred_x = predicted[j].x;
      long cur_pred_y = predicted[j].y;
      long distance = sqrt(pow(cur_obs_x - cur_pred_x, 2) + pow(cur_obs_y - cur_pred_y, 2));
      if (min_distance == -1 || distance < min_distance) {
        min_distance = distance;
        cur_id = predicted[j].id;
      }
    }
    
    observations[i].id = cur_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double weight_normalizer = 0.0;
  for (int i = 0; i < num_particles; i++) {
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
    
    // step 1 transform coordinate
    vector<LandmarkObs> observations_map_coordinator;
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs obs_transformed;
      obs_transformed.id = observations[j].id;
      obs_transformed.x = particle_x + cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y;
      obs_transformed.y = particle_y + sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y;
      observations_map_coordinator.push_back(obs_transformed);
    }
    
    /*Step 2: Filter map landmarks to keep only those which are in the sensor_range of current 
     particle. Push them to predictions vector.*/
    vector<LandmarkObs> predicted_landmarks;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];
      if ((fabs((particle_x - current_landmark.x_f)) <= sensor_range)
          && (fabs((particle_y - current_landmark.y_f)) <= sensor_range)) {
        predicted_landmarks.push_back(LandmarkObs {current_landmark.id_i, current_landmark.x_f, current_landmark.y_f});
      }
    }
    
    // step 3 dataAssociation
    dataAssociation(predicted_landmarks, observations_map_coordinator);
    
    // step 4 calc weights
    particles[i].weight = 1.0;
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double normalizer = 1.0 / (2.0 * M_PI * sig_x * sig_y);
    
    for (int j = 0; j < observations_map_coordinator.size(); j++) {
      double obs_x = observations_map_coordinator[j].x;
      double obs_y = observations_map_coordinator[j].y;
      
      for (int k = 0; k < predicted_landmarks.size(); k++) {
        if (observations_map_coordinator[j].id == predicted_landmarks[k].id) {
          double pred_x = predicted_landmarks[k].x;
          double pred_y = predicted_landmarks[k].y;
          double weight = normalizer * exp(-(pow(pred_x - obs_x, 2) / (2.0 * pow(sig_x, 2))
                                         + pow(pred_y - obs_y, 2) / (2.0 * pow(sig_y, 2))));
          particles[i].weight *= weight;
        }
      }
    }
    
    weight_normalizer += particles[i].weight;
  }
  
  // normalizer particles weight
  for (int i = 0; i < num_particles; i++) {
    particles[i].weight /= weight_normalizer;
    weights[i] = particles[i].weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::default_random_engine gen;
  vector<Particle> resampled_paticles;
  
  std::uniform_int_distribution<int> random_indx_gen(0, num_particles - 1);
  
  int current_index = random_indx_gen(gen);
  double beta = 0.0;
  double max_weight_2 = 2.0 * *max_element(weights.begin(), weights.end());
  
  std::uniform_real_distribution<double> random_beta_gen(0.0, max_weight_2);
  
  for (int i = 0; i < num_particles; i++) {
    beta += random_beta_gen(gen);
    while (beta > weights[current_index]) {
      beta -= particles[current_index].weight;
      current_index = (current_index + 1) % num_particles;
    } 
    
    resampled_paticles.push_back(particles[current_index]);
  }
  
  particles = resampled_paticles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
