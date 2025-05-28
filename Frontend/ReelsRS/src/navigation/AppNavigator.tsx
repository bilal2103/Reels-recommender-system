import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';

// Import screens
import LoginScreen from '../screens/LoginScreen';
import PreferencesScreen from '../screens/PreferencesScreen';
import ReelsScreen from '../screens/ReelsScreen';

// Define the types for our stack navigator
export type RootStackParamList = {
  Login: undefined;
  Preferences: { userId: string };
  Home: undefined;
  Reels: { userId: string; initialReelId: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Login"
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: 'transparent' },
        }}
      >
        <Stack.Screen name="Login" component={LoginScreen} />
        <Stack.Screen name="Preferences" component={PreferencesScreen} />
        <Stack.Screen name="Reels" component={ReelsScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator; 