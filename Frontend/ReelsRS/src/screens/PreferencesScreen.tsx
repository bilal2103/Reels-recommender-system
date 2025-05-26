import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  StatusBar,
  ImageBackground,
  SafeAreaView,
  Alert,
  ActivityIndicator,
} from 'react-native';
import axios from 'axios';
import api from '../services/api';

const PREFERENCE_OPTIONS = ['Gym', 'Cars', 'Food', 'Gaming', 'Oddly Satisfying'];
const MAX_SELECTIONS = 3;

const PreferencesScreen = ({ route, navigation }: any) => {
  const { userId } = route.params || {};
  const [selectedPreferences, setSelectedPreferences] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const togglePreference = (preference: string) => {
    if (selectedPreferences.includes(preference)) {
      // Remove if already selected
      setSelectedPreferences(selectedPreferences.filter(item => item !== preference));
    } else if (selectedPreferences.length < MAX_SELECTIONS) {
      // Add if limit not reached
      setSelectedPreferences([...selectedPreferences, preference]);
    } else {
      // Alert if trying to select more than the limit
      Alert.alert('Selection Limit', `Please select only ${MAX_SELECTIONS} preferences.`);
    }
  };

  const isPreferenceSelected = (preference: string) => {
    return selectedPreferences.includes(preference);
  };

  const savePreferences = async () => {
    if (selectedPreferences.length !== MAX_SELECTIONS) {
      Alert.alert('Selection Required', `Please select exactly ${MAX_SELECTIONS} preferences.`);
      return;
    }

    setIsLoading(true);

    try {
      // Send parameters in the request body
      const requestData = {
        userId: userId,
        initialPreferences: selectedPreferences
      };
      
      console.log('Sending request with data:', requestData);
      
      const response = await api.patch('/set-initial-preferences', requestData);

      if (response.data.success) {
        Alert.alert('Success', 'Your preferences have been saved!', [
          {
            text: 'Continue',
            onPress: () => {
              // Navigate to Reels screen with initial reels and userId
              navigation.navigate('Reels', {
                initialPreferenceReelId: response.data.initialPreferenceReelId,
                userId: userId
              });
            },
          }
        ]);
      } else {
        Alert.alert('Error', 'Failed to save preferences. Please try again.');
      }
    } catch (error) {
      console.error('Save preferences error:', error);
      Alert.alert('Error', 'Something went wrong. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar translucent backgroundColor="transparent" />
      <ImageBackground
        source={{ uri: 'https://images.unsplash.com/photo-1592659762303-90081d34b277?q=80&w=1473&auto=format&fit=crop' }}
        style={styles.backgroundImage}
      >
        <View style={styles.overlay}>
          <View style={styles.headerContainer}>
            <Text style={styles.headerTitle}>Choose Your Interests</Text>
            <Text style={styles.headerSubtitle}>
              Select {MAX_SELECTIONS} categories to personalize your feed
            </Text>
          </View>

          <View style={styles.preferencesContainer}>
            {PREFERENCE_OPTIONS.map((preference) => (
              <TouchableOpacity
                key={preference}
                style={[
                  styles.preferenceItem,
                  isPreferenceSelected(preference) && styles.preferenceItemSelected,
                ]}
                onPress={() => togglePreference(preference)}
                disabled={isLoading}
              >
                <Text
                  style={[
                    styles.preferenceText,
                    isPreferenceSelected(preference) && styles.preferenceTextSelected,
                  ]}
                >
                  {preference}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <View style={styles.selectionInfo}>
            <Text style={styles.selectionInfoText}>
              {selectedPreferences.length} of {MAX_SELECTIONS} selected
            </Text>
          </View>

          <TouchableOpacity
            style={[
              styles.continueButton,
              (selectedPreferences.length !== MAX_SELECTIONS || isLoading) && 
                styles.continueButtonDisabled,
            ]}
            onPress={savePreferences}
            disabled={selectedPreferences.length !== MAX_SELECTIONS || isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="#ffffff" size="small" />
            ) : (
              <Text style={styles.continueButtonText}>Continue</Text>
            )}
          </TouchableOpacity>
        </View>
      </ImageBackground>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  backgroundImage: {
    flex: 1,
    resizeMode: 'cover',
  },
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 20,
    justifyContent: 'center',
  },
  headerContainer: {
    marginBottom: 40,
    alignItems: 'center',
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 10,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#e0e0e0',
    textAlign: 'center',
  },
  preferencesContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginBottom: 30,
  },
  preferenceItem: {
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    paddingVertical: 12,
    paddingHorizontal: 20,
    margin: 8,
    borderRadius: 25,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  preferenceItemSelected: {
    backgroundColor: '#ff4d67',
    borderColor: '#ff4d67',
  },
  preferenceText: {
    color: '#ffffff',
    fontSize: 16,
  },
  preferenceTextSelected: {
    fontWeight: 'bold',
  },
  selectionInfo: {
    alignItems: 'center',
    marginBottom: 30,
  },
  selectionInfoText: {
    color: '#e0e0e0',
    fontSize: 14,
  },
  continueButton: {
    backgroundColor: '#ff4d67',
    paddingVertical: 15,
    paddingHorizontal: 20,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 40,
  },
  continueButtonDisabled: {
    backgroundColor: 'rgba(255, 77, 103, 0.5)',
  },
  continueButtonText: {
    color: '#ffffff',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default PreferencesScreen; 