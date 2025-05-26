import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ImageBackground,
  KeyboardAvoidingView,
  Platform,
  StatusBar,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { authAPI } from '../services/api';

const LoginScreen = ({ navigation }: any) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const validateForm = () => {
    if (!email.trim()) {
      setErrorMessage('Email is required');
      return false;
    }
    if (!password.trim()) {
      setErrorMessage('Password is required');
      return false;
    }
    return true;
  };

  const handleLogin = async () => {
    setErrorMessage('');
    
    if (!validateForm()) {
      return;
    }
    
    setIsLoading(true);
    
    try {
      const response = await authAPI.login(email, password);
      console.log('Login successful:', response);
      
      // Handle successful login
      if (response.id) {
        // Store user ID or token for session management
        if (!response.existingUser) {
          // Navigate to preferences screen for new users
          navigation.navigate('Preferences', { userId: response.id });
        } else {
          // For existing users, navigate to home screen
          Alert.alert('Welcome back!');
          // navigation.navigate('Home');
        }
      } else {
        setErrorMessage('Login failed. Please try again.');
      }
    } catch (error: any) {
      console.error('Login error:', error);
      setErrorMessage(
        error.response?.data?.error || 
        'Failed to connect to the server. Please check your internet connection.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <StatusBar translucent backgroundColor="transparent" />
      <ImageBackground
        source={{ uri: 'https://images.unsplash.com/photo-1616763355548-1b606f439f86?q=80&w=1470&auto=format&fit=crop' }}
        style={styles.backgroundImage}
      >
        <KeyboardAvoidingView
          behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
          style={styles.formContainer}
        >
          <View style={styles.overlay}>
            <Text style={styles.logo}>ReelsRS</Text>
            <Text style={styles.tagline}>Experience short videos like never before</Text>
            
            {errorMessage ? (
              <View style={styles.errorContainer}>
                <Text style={styles.errorText}>{errorMessage}</Text>
              </View>
            ) : null}
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Email or Username"
                placeholderTextColor="#8e8e8e"
                value={email}
                onChangeText={(text) => {
                  setEmail(text);
                  setErrorMessage('');
                }}
                autoCapitalize="none"
                keyboardType="email-address"
                editable={!isLoading}
              />
            </View>
            
            <View style={styles.inputContainer}>
              <TextInput
                style={styles.input}
                placeholder="Password"
                placeholderTextColor="#8e8e8e"
                value={password}
                onChangeText={(text) => {
                  setPassword(text);
                  setErrorMessage('');
                }}
                secureTextEntry
                editable={!isLoading}
              />
            </View>
            
            <TouchableOpacity 
              style={[styles.loginButton, isLoading && styles.loginButtonDisabled]}
              onPress={handleLogin}
              disabled={isLoading}
            >
              {isLoading ? (
                <ActivityIndicator color="#ffffff" size="small" />
              ) : (
                <Text style={styles.loginButtonText}>Log In</Text>
              )}
            </TouchableOpacity>
            
            <TouchableOpacity style={styles.forgotPassword} disabled={isLoading}>
              <Text style={styles.forgotPasswordText}>Forgot password?</Text>
            </TouchableOpacity>
            
            <View style={styles.divider}>
              <View style={styles.dividerLine} />
              <Text style={styles.dividerText}>OR</Text>
              <View style={styles.dividerLine} />
            </View>
            
            <TouchableOpacity style={styles.signupContainer} disabled={isLoading}>
              <Text style={styles.signupText}>
                Don't have an account? <Text style={styles.signupLink}>Sign up</Text>
              </Text>
            </TouchableOpacity>
          </View>
        </KeyboardAvoidingView>
      </ImageBackground>
    </View>
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
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    borderRadius: 20,
    padding: 20,
    width: '90%',
    alignSelf: 'center',
  },
  formContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  logo: {
    fontSize: 42,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 10,
  },
  tagline: {
    fontSize: 16,
    color: '#e0e0e0',
    textAlign: 'center',
    marginBottom: 40,
  },
  errorContainer: {
    backgroundColor: 'rgba(255, 87, 87, 0.2)',
    padding: 10,
    borderRadius: 5,
    marginBottom: 15,
  },
  errorText: {
    color: '#ff5757',
    textAlign: 'center',
  },
  inputContainer: {
    width: '100%',
    marginBottom: 15,
  },
  input: {
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    color: '#ffffff',
    borderRadius: 8,
    padding: 15,
    fontSize: 16,
  },
  loginButton: {
    backgroundColor: '#ff4d67',
    padding: 15,
    borderRadius: 8,
    width: '100%',
    marginTop: 10,
    alignItems: 'center',
  },
  loginButtonDisabled: {
    backgroundColor: '#ff4d6780',
  },
  loginButtonText: {
    color: '#ffffff',
    fontWeight: 'bold',
    textAlign: 'center',
    fontSize: 16,
  },
  forgotPassword: {
    marginTop: 15,
    alignSelf: 'center',
  },
  forgotPasswordText: {
    color: '#e0e0e0',
    fontSize: 14,
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 30,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
  },
  dividerText: {
    color: '#e0e0e0',
    marginHorizontal: 10,
  },
  signupContainer: {
    alignSelf: 'center',
  },
  signupText: {
    color: '#e0e0e0',
    fontSize: 14,
  },
  signupLink: {
    color: '#ff4d67',
    fontWeight: 'bold',
  },
});

export default LoginScreen; 