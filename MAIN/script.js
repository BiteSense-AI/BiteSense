let model
window.onload = () => {
    model = new KerasJS.Model({
        filepath:"https://transcranial.github.io/keras-js-demos-data/resnet50/resnet50.bin"
    })
    model.ready().then(()=>{
        console.log("readu")
    })
}

function predict(){
    const ctx = document.getElementById('c').getContext('2d')
    const imageData = ctx.getImageData(0, 0, 224, 224)
  // model.predict(inputData).then(outputData => {
  //   //this.inferenceTime = this.model.predictStats.forwardPass
  //   this.output = outputData[outputName]
  //   //this.modelRunning = false
  //   //this.updateVis(this.outputClasses[0].index)
  // })
}

document.getElementById("upload").onclick = () => {
    const reader = new FileReader();
    reader.addEventListener(
        "load",
        () => {
          // convert image file to base64 string
          let base_image = new Image();
        //console.log(reader.result)
        base_image.src = reader.result;
        document.getElementById('c').getContext('2d').drawImage(base_image, 0, 0,224,224);
        document.getElementById('c').toBlob((blob) => {
          
          const formData = new FormData();
          formData.append('image', blob);

          fetch('/classify', {
            method: "POST",
            body: formData,
          }).then((res)=>{
            if (res.status === 200) {
              const text = res.text();
              console.log(text)
            } else {
              console.log('classify error')
            }
            })
          })

        
        //console.log("f")
        },
        false,
      );
      //console.log(document.getElementById('f').files[0])
      reader.readAsDataURL(document.getElementById('f').files[0]);
      predict()
}


// var time = 15

// const constraints = {
//   video: {
//     facingMode: "user",
//     width: {
//       min: 1280,
//       ideal: 1920,
//       max: 2560,
//     },
//     height: {
//       min: 720,
//       ideal: 1080,
//       max: 1440,
//     },
//   },
// };

// let canvas;
// window.onload = () => {
//   document.querySelector("video").style.marginLeft =
//     -(document.querySelector("video").offsetWidth - 500) / 2;
//   if ("mediaDevices" in navigator && navigator.mediaDevices.getUserMedia) {
//     startStream(constraints);
//   }
//   canvas = document.getElementById("c");
// };

// window.onresize = () => {
//   document.querySelector("video").style.marginLeft = -(document.querySelector("video").offsetWidth - 500) / 2;
// };

// const video = document.querySelector("video");
// const startStream = async (constraints) => {
//   let stream = await navigator.mediaDevices.getUserMedia(constraints);
//   video.srcObject = stream;
//   init();
// };

// let aModel,
//   nModel,
//   webcam,
//   labelContainer,
//   aMaxPredictions,
//   nMaxPredictions,
//   isAlpha;
// async function init() {
//   const NumberURL = "https://teachablemachine.withgoogle.com/models/33Jw6otUj/";
//   const nModelURL = NumberURL + "model.json";
//   const nMetadataURL = NumberURL + "metadata.json";

//   nModel = await tmImage.load(nModelURL, nMetadataURL);
//   nMaxPredictions = nModel.getTotalClasses();

//   isAlpha = true;
//   window.requestAnimationFrame(loop);
// }

// async function loop() {
//   canvas.width = 500;
//   canvas.height = 500;
//   canvas.getContext("2d").translate(500, 0);
//   canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
//   canvas.getContext("2d").scale(-1, 1);
//   canvas
//     .getContext("2d")
//     .drawImage(
//       video,
//       (document.querySelector("video").offsetWidth - 80) / 2,
//       0,
//       720,
//       720,
//       0,
//       0,
//       500,
//       500
//     );
//   await predict();
//   window.requestAnimationFrame(loop);
// }
// let lastLetter, numIters
// async function predict() {
//   try {
//     const prediction = await (isAlpha ? aModel : nModel).predict(canvas);
//     let highest = -1,
//       classPrediction = "";
//     for (let i = 0; i < (isAlpha ? aMaxPredictions : nMaxPredictions); i++) {
//       if (prediction[i].probability > highest) {
//         highest = prediction[i].probability;
//         classPrediction = prediction[i].className;
//       }
//     }
//     console.log(classPrediction)
//   } catch (e) {
//     console.log(e);
//   }
// }