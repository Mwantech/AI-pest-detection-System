import React, { useState, useRef } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ActivityIndicator,
  ScrollView,
  Alert,
  Dimensions,
  Platform,
  StatusBar,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Card, ProgressBar, IconButton } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

const { width, height } = Dimensions.get('window');
const API_URL = 'http://your-backend-url:3000/api/detect-pest';
const HISTORY_KEY = '@pest_detection_history';

const PestDetectionScreen = () => {
  const navigation = useNavigation();
  const insets = useSafeAreaInsets();
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [pestInfo, setPestInfo] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [facing, setFacing] = useState('back');
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);

  const saveToHistory = async (data) => {
    try {
      const timestamp = new Date().toISOString();
      const historyItem = {
        id: timestamp,
        image: image,
        predictions: predictions,
        pestInfo: pestInfo,
        timestamp: timestamp,
      };

      const existingHistory = await AsyncStorage.getItem(HISTORY_KEY);
      const history = existingHistory ? JSON.parse(existingHistory) : [];
      history.unshift(historyItem);
      
      const trimmedHistory = history.slice(0, 50);
      await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(trimmedHistory));
    } catch (error) {
      console.error('Error saving to history:', error);
    }
  };

  const handleShowCamera = async () => {
    if (!permission?.granted) {
      const permissionResult = await requestPermission();
      if (!permissionResult.granted) {
        Alert.alert('Permission denied', 'Camera permission is required to take photos');
        return;
      }
    }
    setShowCamera(true);
  };

  const handleTakePicture = async () => {
    if (!cameraRef.current) {
      Alert.alert('Error', 'Camera not ready');
      return;
    }

    try {
      const photo = await cameraRef.current.takePictureAsync();
      setImage(photo.uri);
      setShowCamera(false);
      analyzePestImage(photo.uri);
    } catch (error) {
      console.error('Error taking picture:', error);
      Alert.alert('Error', 'Failed to take picture');
    }
  };

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const handleBack = () => {
    setShowCamera(false);
  };

  const pickImage = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      analyzePestImage(result.assets[0].uri);
    }
  };

  const analyzePestImage = async (imageUri) => {
    try {
      setLoading(true);
      setPredictions(null);
      setPestInfo(null);

      const formData = new FormData();
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'pest_image.jpg',
      });

      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPredictions(response.data.predictions);
      setPestInfo(response.data.pest_info);
      await saveToHistory({
        predictions: response.data.predictions,
        pestInfo: response.data.pest_info,
      });
    } catch (error) {
      Alert.alert(
        'Error',
        error.response?.data?.error || 'Failed to analyze image'
      );
    } finally {
      setLoading(false);
    }
  };

  if (showCamera) {
    return (
      <View style={styles.container}>
        <StatusBar barStyle="light-content" />
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          onMountError={(error) => {
            console.error('Camera mount error:', error);
            Alert.alert('Error', 'Failed to start camera');
            setShowCamera(false);
          }}
        >
          <BlurView intensity={80} style={styles.cameraHeader}>
            <TouchableOpacity
              style={styles.backButton}
              onPress={handleBack}
            >
              <IconButton icon="arrow-left" size={24} color="white" />
            </TouchableOpacity>
          </BlurView>

          <View style={styles.cameraOverlay}>
            <View style={styles.focusFrame} />
          </View>

          <BlurView intensity={80} style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.flipButton}
              onPress={toggleCameraFacing}
            >
              <IconButton icon="camera-flip" size={24} color="white" />
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.captureButton}
              onPress={handleTakePicture}
            >
              <View style={styles.captureButtonInner}>
                <View style={styles.captureButtonCore} />
              </View>
            </TouchableOpacity>

            <TouchableOpacity style={styles.flipButton}>
              <IconButton icon="flash" size={24} color="white" />
            </TouchableOpacity>
          </BlurView>
        </CameraView>
      </View>
    );
  }

  const renderPredictions = () => {
    if (!predictions) return null;

    return predictions.map((pred, index) => (
      <Card key={index} style={styles.predictionCard}>
        <LinearGradient
          colors={['#ffffff', '#f8f9fa']}
          style={styles.predictionGradient}
        >
          <Card.Content>
            <View style={styles.predictionRow}>
              <Text style={styles.predictionText}>{pred.class}</Text>
              <View style={styles.confidenceBadge}>
                <Text style={styles.confidenceText}>{pred.confidence.toFixed(1)}%</Text>
              </View>
            </View>
            <ProgressBar
              progress={pred.confidence / 100}
              color="#2196F3"
              style={styles.progressBar}
            />
          </Card.Content>
        </LinearGradient>
      </Card>
    ));
  };

  const renderPestInfo = () => {
    if (!pestInfo) return null;

    return (
      <Card style={styles.pestInfoCard}>
        <LinearGradient
          colors={['#ffffff', '#f8f9fa']}
          style={styles.pestInfoGradient}
        >
          <Card.Content>
            <Text style={styles.pestInfoTitle}>Pest Information</Text>
            <Text style={styles.scientificName}>
              Scientific Name: {pestInfo.scientific_name}
            </Text>
            
            {['Recommendations', 'Control Measures', 'Recommended Pesticides'].map((section, idx) => (
              <View key={idx} style={styles.infoSection}>
                <Text style={styles.sectionTitle}>{section}</Text>
                {pestInfo[section.toLowerCase().replace(' ', '_')].map((item, index) => (
                  <View key={index} style={styles.listItemContainer}>
                    <View style={styles.bullet} />
                    <Text style={styles.listItem}>{item}</Text>
                  </View>
                ))}
              </View>
            ))}
          </Card.Content>
        </LinearGradient>
      </Card>
    );
  };

  return (
    <SafeAreaView style={styles.mainContainer}>
      <StatusBar barStyle="dark-content" />
      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingBottom: insets.bottom + 100 } // Add extra padding for tab bar
        ]}
        showsVerticalScrollIndicator={false}
        bounces={true}
      >
        <View style={styles.content}>
          <View style={styles.header}>
            <Text style={styles.title}>Pest Detection</Text>
            <TouchableOpacity
              style={styles.historyButton}
              onPress={() => navigation.navigate('History')}
            >
              <IconButton icon="history" size={24} color="#2196F3" />
            </TouchableOpacity>
          </View>

          <View style={styles.buttonContainer}>
            <TouchableOpacity
              style={[styles.button, styles.buttonPrimary]}
              onPress={handleShowCamera}
            >
              <LinearGradient
                colors={['#2196F3', '#1976D2']}
                style={styles.buttonGradient}
              >
                <IconButton icon="camera" size={24} color="white" />
                <Text style={styles.buttonText}>Take Photo</Text>
              </LinearGradient>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.button, styles.buttonSecondary]}
              onPress={pickImage}
            >
              <LinearGradient
                colors={['#4CAF50', '#388E3C']}
                style={styles.buttonGradient}
              >
                <IconButton icon="image" size={24} color="white" />
                <Text style={styles.buttonText}>Gallery</Text>
              </LinearGradient>
            </TouchableOpacity>
          </View>

          {image && (
            <Card style={styles.imageCard}>
              <Card.Cover 
                source={{ uri: image }} 
                style={styles.selectedImage}
                resizeMode="cover"
              />
            </Card>
          )}

          {loading && (
            <View style={styles.loaderContainer}>
              <ActivityIndicator size="large" color="#2196F3" />
              <Text style={styles.loaderText}>Analyzing image...</Text>
            </View>
          )}

          {/* Predictions and pest info rendering remain the same */}
          {renderPredictions()}
          {renderPestInfo()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  mainContainer: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  content: {
    padding: 20,
  },
  container: {
    flex: 1,
    backgroundColor: '#ffffff',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
    paddingTop: Platform.OS === 'android' ? StatusBar.currentHeight : 0,
  },
  title: {
    fontSize: 32,
    fontWeight: '800',
    color: '#1a1a1a',
    letterSpacing: -0.5,
  },
  historyButton: {
    backgroundColor: '#f5f5f5',
    borderRadius: 12,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  button: {
    width: '48%',
    height: 56,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  buttonGradient: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 12,
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  camera: {
    flex: 1,
    backgroundColor: '#000',
  },
  cameraHeader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1,
    padding: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  focusFrame: {
    width: width * 0.8,
    height: width * 0.8,
    borderWidth: 2,
    borderColor: 'rgba(255, 255, 255, 0.5)',
    borderRadius: 20,
  },
  cameraControls: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
    paddingBottom: 40,
  },
  captureButton: {
    width: 80,
    height: 80,
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonCore: {
    width: 54,
    height: 54,
    borderRadius: 27,
    backgroundColor: '#fff',
  },
  imageCard: {
    marginVertical: 24,
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  selectedImage: {
    height: 300,
    backgroundColor: '#f5f5f5',
  },
  loaderContainer: {
    alignItems: 'center',
    padding: 24,
  },
  loaderText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  predictionCard: {
    marginVertical: 8,
    borderRadius: 16,
    overflow: 'hidden',
    elevation: 2,
  },
  predictionGradient: {
    padding: 16,
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  predictionText: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a1a',
  },
  confidenceBadge: {
    backgroundColor: '#e3f2fd',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 12,
  },
  confidenceText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2196F3',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  pestInfoCard: {
    marginTop: 24,
    borderRadius: 20,
    overflow: 'hidden',
    elevation: 3,
  },
  pestInfoGradient: {
    padding: 20,
  },
  pestInfoTitle: {
    fontSize: 24,
    fontWeight: '800',
    color: '#1a1a1a',
    marginBottom: 16,
  },
  scientificName: {
    fontSize: 16,
    fontStyle: 'italic',
    color: '#666',
    marginBottom: 20,
  },
  infoSection: {
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1a1a1a',
    marginBottom: 12,
  },
  listItemContainer: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 8,
    paddingRight: 16,
  },
  bullet: {
    width: 6,
    height: 6,
    borderRadius: 3,
    backgroundColor: '#2196F3',
    marginTop: 8,
    marginRight: 12,
  },
  listItem: {
    flex: 1,
    fontSize: 16,
    lineHeight: 22,
    color: '#333',
  },
});

export default PestDetectionScreen;