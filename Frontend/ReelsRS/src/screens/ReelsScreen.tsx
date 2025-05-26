import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  TouchableOpacity,
  Image,
  SafeAreaView,
  ActivityIndicator,
  PanResponder,
  Animated,
} from 'react-native';
import Video from 'react-native-video';
import MaterialCommunityIcons from 'react-native-vector-icons/MaterialCommunityIcons';
import { reelsAPI } from '../services/api';

const { width, height } = Dimensions.get('window');

interface Reel {
  category: string;
  video_name: string;
  reel_id: string;
}

const ReelsScreen = ({ route }: any) => {
  const [userId] = useState(route.params?.userId || '');
  const [currentReelId, setCurrentReelId] = useState(route.params?.initialPreferenceReelId || '');
  const [nextReelId, setNextReelId] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [isPaused, setIsPaused] = useState(false);
  const [isVideoLoaded, setIsVideoLoaded] = useState(false);
  const [isLiked, setIsLiked] = useState(false);
  const [videoDuration, setVideoDuration] = useState(0);
  const [reelCounter, setReelCounter] = useState(0);
  const watchTimeRef = useRef(0);
  const position = new Animated.Value(0);
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Log initial props and state
  useEffect(() => {
    console.log('=== ReelsScreen Mounted ===');
    console.log('Initial Route Params:', route.params);
    console.log('User ID:', userId);
    console.log('Initial Reel ID:', currentReelId);

    // Load initial reel content
    if (currentReelId) {
      loadReelContent(currentReelId, true);
    }
  }, []);

  const loadReelContent = async (reelId: string, isCurrent: boolean = true) => {
    try {
      console.log(`Loading ${isCurrent ? 'current' : 'next'} reel content for:`, reelId);
      
      if (isCurrent) {
        setIsLoading(true);
        setIsVideoLoaded(false);
      }

      // We don't need to fetch the content separately since we're using the URL directly in Video component
      if (isCurrent) {
        setIsLoading(false);
      }
    } catch (error) {
      console.error('Error loading reel content:', error);
      if (isCurrent) {
        setIsLoading(false);
        setIsVideoLoaded(false);
      }
    }
  };

  const preloadNextReel = async () => {
    try {
      console.log('Preloading next reel...');
      const nextReelCount = reelCounter + 1;
      let nextReel;
      
      // Every fourth reel (1-based counting), use serendipity API
      if (nextReelCount % 4 === 0) {
        console.log('Getting serendipity reel...');
        const serendipityResponse = await reelsAPI.getSerendipityReel(userId, currentReelId);
        nextReel = serendipityResponse;  // The API already returns the reelId directly
      } else {
        nextReel = await reelsAPI.getSimilarReels(userId, currentReelId);
      }
      
      console.log('Next reel ID:', nextReel);
      if (nextReel) {
        setNextReelId(nextReel);
        await loadReelContent(nextReel, false);
      }
    } catch (error) {
      console.error('Error preloading next reel:', error);
    }
  };

  const switchToNextReel = async () => {
    console.log('Switching to next reel');
    if (nextReelId && !isTransitioning) {
      setIsTransitioning(true);
      setIsLoading(true);
      
      try {
        // Mark current reel as watched
        await markCurrentReelAsWatched();
        
        // Reset states for new reel
        setIsLiked(false);
        watchTimeRef.current = 0;
        setVideoDuration(0);
        
        // Switch reels
        const previousReelId = currentReelId;
        setCurrentReelId(nextReelId);
        
        // Increment reel counter and preload next reel
        setReelCounter(prevCounter => {
          const newCounter = prevCounter + 1;
          // Start preloading the next reel immediately
          const nextReelCount = newCounter + 1; // Check for the next reel
          
          // Preload next reel asynchronously
          (async () => {
            try {
              let newNextReel;
              if (nextReelCount % 4 === 0) {
                const serendipityResponse = await reelsAPI.getSerendipityReel(userId, nextReelId);
                newNextReel = serendipityResponse;  // The API already returns the reelId directly
              } else {
                newNextReel = await reelsAPI.getSimilarReels(userId, nextReelId);
              }
              
              if (newNextReel) {
                setNextReelId(newNextReel);
                loadReelContent(newNextReel, false);
              }
            } catch (error) {
              console.error('Error preloading next reel:', error);
            }
          })();
          
          return newCounter;
        });
        
        setNextReelId('');
        setIsVideoLoaded(false);
      } catch (error) {
        console.error('Error during reel switch:', error);
      } finally {
        setIsLoading(false);
        setIsTransitioning(false);
        // Reset position with animation
        Animated.spring(position, {
          toValue: 0,
          useNativeDriver: true,
          tension: 40,
          friction: 7
        }).start();
      }
    }
  };

  const panResponder = PanResponder.create({
    onStartShouldSetPanResponder: () => true,
    onMoveShouldSetPanResponder: () => true,
    onPanResponderMove: (_, gestureState) => {
      if (!isTransitioning) {
        position.setValue(gestureState.dy);
      }
    },
    onPanResponderRelease: async (_, gestureState) => {
      if (gestureState.dy < -50 && nextReelId && !isTransitioning) {
        // Animate to final position before switching
        Animated.timing(position, {
          toValue: -height,
          duration: 200,
          useNativeDriver: true
        }).start(async () => {
          await switchToNextReel();
        });
      } else {
        // Reset position if not switching
        Animated.spring(position, {
          toValue: 0,
          useNativeDriver: true,
          tension: 40,
          friction: 7
        }).start();
      }
    },
  });

  const calculateRating = () => {
    if (isLiked) return 5;
    
    if (videoDuration === 0) return 1;
    
    const watchPercentage = (watchTimeRef.current / videoDuration) * 100;
    console.log('Watch time:', watchTimeRef.current, 'Duration:', videoDuration, 'Percentage:', watchPercentage);
    
    // Calculate rating based on watch percentage
    if (watchPercentage >= 80) return 5;
    if (watchPercentage >= 60) return 4;
    if (watchPercentage >= 40) return 3;
    if (watchPercentage >= 20) return 2;
    return 1;
  };

  const markCurrentReelAsWatched = async () => {
    try {
      const rating = calculateRating();
      await reelsAPI.markAsWatched(userId, currentReelId, rating);
      console.log('Marked as watched:', currentReelId, 'with rating:', rating);
    } catch (error) {
      console.error('Error marking reel as watched:', error);
    }
  };

  const handleVideoLoad = (response: any) => {
    console.log('Video loaded successfully with duration:', response.duration);
    setVideoDuration(response.duration);
    setIsVideoLoaded(true);
    setIsLoading(false);
    watchTimeRef.current = 0;
    
    // Start preloading next reel when current one is loaded
    if (!nextReelId) {
      preloadNextReel();
    }
  };

  const handleVideoProgress = (progress: any) => {
    if (!isPaused && progress.currentTime) {
      watchTimeRef.current = progress.currentTime;
      console.log('Current watch time:', watchTimeRef.current);
    }
  };

  const handleLikePress = async () => {
    setIsLiked(!isLiked);
    if (!isLiked) {
      // If user is liking the video, immediately mark it as watched with 5-star rating
      try {
        await reelsAPI.markAsWatched(userId, currentReelId, 5);
        console.log('Marked as liked with 5-star rating:', currentReelId);
      } catch (error) {
        console.error('Error marking reel as liked:', error);
      }
    }
  };

  const handleVideoEnd = async () => {
    console.log('Video ended');
    if (nextReelId) {
      await switchToNextReel();
    }
  };

  const handleVideoError = (error: any) => {
    console.error('Video playback error:', error);
    setIsLoading(false);
    setIsVideoLoaded(false);
  };

  const renderVideo = () => {
    const handleVideoPress = () => {
      console.log('Video pressed, toggling pause state');
      setIsPaused(!isPaused);
    };

    return (
      <TouchableOpacity 
        activeOpacity={1} 
        onPress={handleVideoPress}
        style={styles.videoContainer}
      >
        <Video
          source={{ uri: `http://10.0.2.2:8000/api/reel-content?reelId=${currentReelId}` }}
          style={styles.video}
          resizeMode="cover"
          repeat={false}
          paused={isPaused}
          onEnd={handleVideoEnd}
          controls={false}
          onError={handleVideoError}
          onLoad={handleVideoLoad}
          onProgress={handleVideoProgress}
          progressUpdateInterval={500}
        />
        {(!isVideoLoaded || isLoading) && (
          <View style={styles.loadingOverlay}>
            <ActivityIndicator size="large" color="#FF4081" />
            <Text style={styles.loadingText}>Loading reel...</Text>
          </View>
        )}
        {isPaused && isVideoLoaded && (
          <View style={styles.pauseOverlay}>
            <MaterialCommunityIcons name="play-circle" size={64} color="#ffffff" />
          </View>
        )}
      </TouchableOpacity>
    );
  };

  const renderInteractionBar = () => {
    return (
      <View style={styles.interactionBar}>
        <TouchableOpacity style={styles.interactionButton} onPress={handleLikePress}>
          <MaterialCommunityIcons 
            name={isLiked ? "heart" : "heart-outline"} 
            size={32} 
            color={isLiked ? "#FF4081" : "#ffffff"} 
          />
          <Text style={styles.interactionText}>Like</Text>
        </TouchableOpacity>
        
        <TouchableOpacity style={styles.interactionButton}>
          <MaterialCommunityIcons name="share-variant" size={32} color="#ffffff" />
          <Text style={styles.interactionText}>Share</Text>
        </TouchableOpacity>
      </View>
    );
  };

  const renderReelInfo = () => {
    return (
      <View style={styles.reelInfo}>
        <Text style={styles.reelDescription}>
          {/* Empty for now, can be used for other reel info later */}
        </Text>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.container}>
      <Animated.View 
        style={[
          styles.reelContainer,
          {
            transform: [{ translateY: position }]
          }
        ]}
        {...panResponder.panHandlers}
      >
        {renderVideo()}
        {renderInteractionBar()}
        {renderReelInfo()}
      </Animated.View>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  reelContainer: {
    width,
    height: height - 49, // Subtract bottom tab height
    position: 'relative',
    backgroundColor: '#000',
  },
  videoContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  video: {
    width: '100%',
    height: '100%',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#ffffff',
    marginTop: 12,
    fontSize: 16,
  },
  pauseOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  interactionBar: {
    position: 'absolute',
    right: 8,
    bottom: 100,
    alignItems: 'center',
  },
  interactionButton: {
    alignItems: 'center',
    marginVertical: 8,
  },
  interactionText: {
    color: 'white',
    fontSize: 12,
    marginTop: 4,
  },
  reelInfo: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    padding: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  reelDescription: {
    color: 'white',
    fontSize: 14,
    marginBottom: 12,
  },
});

export default ReelsScreen; 