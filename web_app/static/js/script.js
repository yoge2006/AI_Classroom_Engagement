console.log("FocusTrack AI frontend loaded")

// ---------- SIDEBAR TOGGLE ----------

function toggleSidebar(){

const sidebar = document.getElementById("sidebar")
const content = document.querySelector(".content")
const button = document.querySelector(".menu-btn")

if(sidebar && content && button){
sidebar.classList.toggle("active")
content.classList.toggle("shift")
button.classList.toggle("shift")
}

}


// ---------- SOCKET CONNECTION ----------

const socket = io()

// receive attention updates
socket.on("attention_update", function(data){

const scoreElement = document.getElementById("avgScore")
const statusElement = document.getElementById("sessionStatus")
const levelElement = document.getElementById("attentionLevel")

if(scoreElement){
scoreElement.innerText = data.score + "%"
}

// update session status
if(statusElement){

if(data.score > 0){
statusElement.innerText = "Active"
statusElement.className = "attentive"
}else{
statusElement.innerText = "Inactive"
statusElement.className = "distracted"
}

}

// attention level indicator
if(levelElement){

if(data.score >= 80){
levelElement.innerText = "High Attention"
levelElement.style.color = "green"
}
else if(data.score >= 50){
levelElement.innerText = "Moderate Attention"
levelElement.style.color = "orange"
}
else{
levelElement.innerText = "Low Attention"
levelElement.style.color = "red"
}

}

})


// request new data every 2 seconds
setInterval(function(){
socket.emit("request_data")
},2000)



// ---------- SESSION TIMER ----------

let seconds = 0

setInterval(function(){

const statusElement = document.getElementById("sessionStatus")
const timeElement = document.getElementById("sessionTime")

if(statusElement && timeElement){

if(statusElement.innerText === "Active"){

seconds++

let mins = Math.floor(seconds/60)
let secs = seconds % 60

timeElement.innerText =
"Session Time: " + mins + ":" + (secs<10?"0":"") + secs

}

}

},1000)