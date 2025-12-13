let mediaRecorder;
let recordedChunks = [];
let cameraStream = null;

const form = document.getElementById("uploadForm");
const messageDiv = document.getElementById("uploadMessage");
const generateMapBtn = document.getElementById("generateMapBtn");

// Get record and stop buttons
const recordBtn = document.querySelector(".record-btn");
const stopBtn = document.querySelector(".stop-btn");
const connectBtn = document.getElementById("connectBtn");
const disconnectBtn = document.getElementById("disconnectBtn");

// Disable generateMapBtn on page load
generateMapBtn.disabled = true;

// Disable record and stop buttons on page load
recordBtn.disabled = true;
stopBtn.disabled = true;

// UPLOAD VIDEO
form.addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent page reload

    const fileInput = document.getElementById("videoUpload");
    if (fileInput.files.length === 0) {
        alert("Please select a file to upload.");
        return;
    }

    const formData = new FormData();
    formData.append("video", fileInput.files[0]);

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            messageDiv.innerText = "✅ Upload successful!";
            generateMapBtn.disabled = false;
        } else {
            const data = await response.json();
            messageDiv.innerText = "❌ Upload failed. Please try again.";
            generateMapBtn.disabled = true;
        }
    } catch (err) {
        messageDiv.innerText = "❌ Error uploading file: " + err;
        generateMapBtn.disabled = true;
    }
});

//Generate Map
document.getElementById("generateMapBtn").addEventListener("click", function() {
    fetch("/generate_map", {
        method: "POST"
    })
    .then(response => response.json())
    .then(data => {
        alert("Map generation response: " + JSON.stringify(data));
    })
    .catch(err => {
        alert("Error: " + err);
    });
});


// CONNECT TO CAMERA
async function connectToCamera() {
    const camStream = document.getElementById("camStream");
    const camStatus = document.getElementById("camStatus");
    
    try {
        // Request access to the native camera
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false 
        });
        
        // Store camera stream
        cameraStream = stream;
        
        // Create a video element to display the stream
        const video = document.createElement("video");
        video.srcObject = stream;
        video.autoplay = true;
        video.playsinline = true;
        video.style.width = "100%";
        video.style.height = "300px";
        video.style.borderRadius = "12px";
        video.id = "liveVideo";
        video.className = "camera-feed";
        
        // Replace iframe with video element
        camStream.parentElement.replaceChild(video, camStream);
        
        // Align the wrapper properly
        const wrapper = document.querySelector(".map-camera-wrapper");
        wrapper.style.alignItems = "stretch";
        
        // Enable record and stop buttons
        recordBtn.disabled = false;
        stopBtn.disabled = false;
        
        // Toggle button visibility
        connectBtn.style.display = "none";
        disconnectBtn.style.display = "inline-block";
        
        camStatus.innerText = "Status: Camera Connected";
    } catch (err) {
        camStatus.innerText = "Status: Camera Access Denied";
        alert("Camera access denied. Please allow camera access in your browser settings.");
        console.error("Camera error:", err);
    }
}


// DISCONNECT CAMERA
function disconnectCamera() {
    const camStatus = document.getElementById("camStatus");
    
    try {
        // Stop all camera tracks
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        
        // Remove video element and restore iframe
        const liveVideo = document.getElementById("liveVideo");
        if (liveVideo) {
            const camStreamParent = liveVideo.parentElement;
            const newIframe = document.createElement("iframe");
            newIframe.id = "camStream";
            newIframe.className = "camera-feed";
            newIframe.src = "http://192.168.4.1/stream";
            newIframe.setAttribute("allow", "camera");
            
            camStreamParent.replaceChild(newIframe, liveVideo);
        }
        
        // Disable record and stop buttons
        recordBtn.disabled = true;
        stopBtn.disabled = true;
        
        // Toggle button visibility
        connectBtn.style.display = "inline-block";
        disconnectBtn.style.display = "none";
        
        camStatus.innerText = "Status: Disconnected";
    } catch (err) {
        camStatus.innerText = "Status: Disconnect Error";
        console.error("Disconnect error:", err);
    }
}


// RECORDING
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: false });
        mediaRecorder = new MediaRecorder(stream);
        recordedChunks = [];

        mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: "video/webm" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "bimbot_exploration.webm";
            a.click();
        };

        mediaRecorder.start();
        document.getElementById("camStatus").innerText = "Status: Recording...";
    } catch (err) {
        alert("Recording permission denied.");
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
        document.getElementById("camStatus").innerText = "Status: Saved Recording";
    }
}

// LIVE HAZARD DATA FROM ESP32 

async function fetchHazardData() {
    try {
        const res = await fetch("http://192.168.4.1/sensor");
        const data = await res.json();

        const temp = data.temperature;
        const hum = data.humidity;

        const risk = classifyRisk(temp, hum);
        const now = new Date().toLocaleString();

        document.getElementById("hazardStatus").innerText = "Live Hazard Data Detected";
        document.getElementById("tempDisplay").innerText = `Temperature: ${temp} °C`;
        document.getElementById("humDisplay").innerText = `Humidity: ${hum} %`;
        document.getElementById("riskLevel").innerHTML = `<span class="${risk.cls}">${risk.label}</span>`;
        document.getElementById("hazardTime").innerText = `Last Update: ${now}`;

    } catch (err) {
        document.getElementById("hazardStatus").innerText = "No temperature and humidity data detected. Check and turn on the BIM-BOT.";
        document.getElementById("tempDisplay").innerText = "Temperature: -- °C";
        document.getElementById("humDisplay").innerText = "Humidity: -- %";
        document.getElementById("riskLevel").innerHTML = "";
        document.getElementById("hazardTime").innerText = "";
    }
}

setInterval(fetchHazardData, 2000);
