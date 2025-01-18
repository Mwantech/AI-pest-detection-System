import React, { useState } from 'react';
import {
  SafeAreaView,
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  ScrollView,
  Alert,
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { Camera } from 'expo-camera';
import axios from 'axios';
import * as FileSystem from 'expo-file-system';
import { Card, ProgressBar } from 'react-native-paper';

const API_URL = 'http://your-backend-url:3000/api/detect-pest';

const PestDetectionScreen = () => {
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictions, setPredictions] = useState(null);
  const [pestInfo, setPestInfo] = useState(null);
  const [cameraPermission, setCameraPermission] = useState(null);
  const [showCamera, setShowCamera] = useState(false);

  // Request camera permissions
  const requestCameraPermission = async () => {
    const { status } = await Camera.requestCameraPermissionsAsync();
    setCameraPermission(status === 'granted');
    if (status === 'granted') {
      setShowCamera(true);
    } else {
      Alert.alert('Permission denied', 'Camera permission is required to take photos');
    }
  };

  // Pick image from gallery
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

  // Take photo using camera
  const takePhoto = async () => {
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [4, 3],
      quality: 1,
    });

    if (!result.canceled) {
      setImage(result.assets[0].uri);
      setShowCamera(false);
      analyzePestImage(result.assets[0].uri);
    }
  };

  // Analyze pest image
  const analyzePestImage = async (imageUri) => {
    try {
      setLoading(true);
      setPredictions(null);
      setPestInfo(null);

      // Create form data
      const formData = new FormData();
      const imageInfo = await FileSystem.getInfoAsync(imageUri);
      
      formData.append('image', {
        uri: imageUri,
        type: 'image/jpeg',
        name: 'pest_image.jpg',
      });

      // Send to backend
      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setPredictions(response.data.predictions);
      setPestInfo(response.data.pest_info);
    } catch (error) {
      Alert.alert(
        'Error',
        error.response?.data?.error || 'Failed to analyze image'
      );
    } finally {
      setLoading(false);
    }
  };

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

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>Pest Detection</Text>

        <View style={styles.buttonContainer}>
          <TouchableOpacity
            style={styles.button}
            onPress={requestCameraPermission}
          >
            <Text style={styles.buttonText}>Take Photo</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.button}
            onPress={pickImage}
          >
            <Text style={styles.buttonText}>Pick from Gallery</Text>
          </TouchableOpacity>
        </View>

        {image && (
          <Image source={{ uri: image }} style={styles.selectedImage} />
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
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#2196F3',
    padding: 15,
    borderRadius: 8,
    minWidth: 150,
  },
  buttonText: {
    color: 'white',
    textAlign: 'center',
    fontWeight: 'bold',
  },
  selectedImage: {
    width: '100%',
    height: 300,
    resizeMode: 'contain',
    marginVertical: 20,
    borderRadius: 8,
  },
  loader: {
    marginVertical: 20,
  },
  predictionCard: {
    marginVertical: 8,
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  predictionText: {
    fontSize: 16,
    fontWeight: '500',
  },
  confidenceText: {
    fontSize: 16,
    color: '#666',
  },
  progressBar: {
    height: 8,
    borderRadius: 4,
  },
  pestInfoCard: {
    marginTop: 20,
  },
  pestInfoTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  scientificName: {
    fontSize: 16,
    fontStyle: 'italic',
    marginBottom: 15,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    marginTop: 15,
    marginBottom: 8,
  },
  listItem: {
    fontSize: 16,
    marginBottom: 5,
    paddingLeft: 10,
  },
});

export default PestDetectionScreen;


