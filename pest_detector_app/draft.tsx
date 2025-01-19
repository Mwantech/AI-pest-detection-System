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
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as ImagePicker from 'expo-image-picker';
import axios from 'axios';
import { Card, ProgressBar, IconButton } from 'react-native-paper';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';

const API_URL = 'http://your-backend-url:3000/api/detect-pest';
const HISTORY_KEY = '@pest_detection_history';

const PestDetectionScreen = () => {
  const navigation = useNavigation();
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
          <View style={styles.cameraHeader}>
            <TouchableOpacity
              style={styles.backButton}
              onPress={handleBack}
            >
              <Text style={styles.buttonText}>Back</Text>
            </TouchableOpacity>
          </View>

          <View style={styles.cameraControls}>
            <TouchableOpacity
              style={styles.flipButton}
              onPress={toggleCameraFacing}
            >
              <Text style={styles.buttonText}>Flip</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={styles.captureButton}
              onPress={handleTakePicture}
            >
              <View style={styles.captureButtonInner} />
            </TouchableOpacity>

            <View style={styles.flipButton}>
              {/* Empty view for layout balance */}
            </View>
          </View>
        </CameraView>
      </View>
    );
  }


  const renderPredictions = () => {
      if (!predictions) return null;
  
      return predictions.map((pred, index) => (
        <Card key={index} style={styles.predictionCard}>
          <Card.Content>
            <View style={styles.predictionRow}>
              <Text style={styles.predictionText}>{pred.class}</Text>
              <Text style={styles.confidenceText}>{pred.confidence.toFixed(1)}%</Text>
            </View>
            <ProgressBar
              progress={pred.confidence / 100}
              color="#2196F3"
              style={styles.progressBar}
            />
          </Card.Content>
        </Card>
      ));
    };
  
    const renderPestInfo = () => {
      if (!pestInfo) return null;
  
      return (
        <Card style={styles.pestInfoCard}>
          <Card.Content>
            <Text style={styles.pestInfoTitle}>Pest Information</Text>
            <Text style={styles.scientificName}>
              Scientific Name: {pestInfo.scientific_name}
            </Text>
            
            <Text style={styles.sectionTitle}>Recommendations:</Text>
            {pestInfo.recommendations.map((rec, index) => (
              <Text key={index} style={styles.listItem}>• {rec}</Text>
            ))}
  
            <Text style={styles.sectionTitle}>Control Measures:</Text>
            {pestInfo.control_measures.map((measure, index) => (
              <Text key={index} style={styles.listItem}>• {measure}</Text>
            ))}
  
            <Text style={styles.sectionTitle}>Recommended Pesticides:</Text>
            {pestInfo.pesticides.map((pesticide, index) => (
              <Text key={index} style={styles.listItem}>• {pesticide}</Text>
            ))}
          </Card.Content>
        </Card>
      );
    };
  
  // Rest of your render code remains the same...
  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <Text style={styles.title}>Pest Detection</Text>
          <IconButton
            icon="history"
            size={24}
            onPress={() => navigation.navigate('History')}
          />
        </View>

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={[styles.button, styles.buttonPrimary]}
            onPress={handleShowCamera}
          >
            <IconButton icon="camera" size={24} color="white" />
            <Text style={styles.buttonText}>Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.button, styles.buttonSecondary]}
            onPress={pickImage}
          >
            <IconButton icon="image" size={24} color="white" />
            <Text style={styles.buttonText}>Gallery</Text>
          </TouchableOpacity>
        </View>

        {image && (
          <Card style={styles.imageCard}>
            <Card.Cover source={{ uri: image }} style={styles.selectedImage} />
          </Card>
        )}

        {loading && (
          <ActivityIndicator size="large" color="#2196F3" style={styles.loader} />
        )}

        {renderPredictions()}
        {renderPestInfo()}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  scrollContent: {
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1a1a1a',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 15,
    borderRadius: 12,
    width: '48%',
    elevation: 2,
  },
  buttonPrimary: {
    backgroundColor: '#2196F3',
  },
  buttonSecondary: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  camera: {
    flex: 1,
  },
  cameraHeader: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    zIndex: 1,
    padding: 20,
  },
  backButton: {
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    padding: 10,
    borderRadius: 8,
    alignSelf: 'flex-start',
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
  },
  flipButton: {
    padding: 15,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: 8,
    width: 80,
    alignItems: 'center',
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: 'rgba(255, 255, 255, 0.3)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#fff',
  },
  cancelButton: {
    padding: 15,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: 8,
  },
  imageCard: {
    marginVertical: 20,
    borderRadius: 12,
    elevation: 3,
  },
  selectedImage: {
    height: 300,
    borderRadius: 12,
  },
  loader: {
    marginVertical: 20,
  },
  predictionCard: {
    marginVertical: 8,
    borderRadius: 12,
    elevation: 2,
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  predictionText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1a1a1a',
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
    fontWeight: '500',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  pestInfoCard: {
    marginTop: 20,
    borderRadius: 12,
    elevation: 2,
  },
  pestInfoTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#1a1a1a',
    marginBottom: 12,
  },
  scientificName: {
    fontSize: 16,
    fontStyle: 'italic',
    color: '#666',
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1a1a1a',
    marginTop: 15,
    marginBottom: 8,
  },
  listItem: {
    fontSize: 16,
    marginBottom: 6,
    paddingLeft: 10,
    color: '#333',
  },
});

export default PestDetectionScreen;