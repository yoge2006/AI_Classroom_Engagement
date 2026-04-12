console.log("🎯 FocusTrack AI frontend loaded");


// ============================================
// SIDEBAR TOGGLE (Dashboard page)
// ============================================

function toggleSidebar() {
    const sidebar = document.getElementById("sidebar");
    const overlay = document.getElementById("sidebarOverlay");
    const button = document.querySelector(".menu-btn");

    if (sidebar && overlay && button) {
        sidebar.classList.toggle("active");
        overlay.classList.toggle("active");
        button.classList.toggle("shift");
    }
}


// ============================================
// SOCKET CONNECTION
// ============================================

const socket = io();
let peakFacesLocal = 0;

// Receive attention updates (Dashboard page)
socket.on("attention_update", function(data) {

    const scoreElement = document.getElementById("avgScore");
    const statusElement = document.getElementById("sessionStatus");
    const levelElement = document.getElementById("attentionLevel");
    const liveIndicator = document.getElementById("liveIndicator");
    const faceCountEl = document.getElementById("faceCount");
    const peakFacesEl = document.getElementById("peakFaces");
    const summaryScoreEl = document.getElementById("summaryScore");
    const summaryDetailEl = document.getElementById("summaryDetail");

    // Update average score with animation
    if (scoreElement) {
        animateValue(scoreElement, parseInt(scoreElement.innerText) || 0, data.score, 600);
    }

    // Update session status
    if (statusElement) {
        if (data.score > 0) {
            statusElement.innerText = "Active";
            statusElement.style.color = "#22c55e";
        } else {
            statusElement.innerText = "Inactive";
            statusElement.style.color = "#ef4444";
        }
    }

    // Update live indicator badge
    if (liveIndicator) {
        if (data.score > 0 || data.face_count > 0) {
            liveIndicator.className = "status-badge status-active";
            liveIndicator.innerHTML = '<span class="pulse-dot"></span> Monitoring Active';
        }
    }

    // Attention level indicator
    if (levelElement) {
        if (data.score >= 80) {
            levelElement.innerText = "High Attention";
            levelElement.className = "stat-detail attentive";
        } else if (data.score >= 50) {
            levelElement.innerText = "Moderate Attention";
            levelElement.className = "stat-detail moderate";
        } else {
            levelElement.innerText = "Low Attention";
            levelElement.className = "stat-detail distracted";
        }
    }

    // Dynamic face count (Dashboard)
    if (faceCountEl) {
        faceCountEl.innerText = data.face_count || 0;
    }

    // Peak faces
    if (data.peak_faces !== undefined && data.peak_faces > peakFacesLocal) {
        peakFacesLocal = data.peak_faces;
    }

    if (peakFacesEl) {
        peakFacesEl.innerText = "Peak: " + peakFacesLocal + " student" + (peakFacesLocal !== 1 ? "s" : "");
    }

    // Session summary card
    if (summaryScoreEl) {
        summaryScoreEl.innerText = data.score + "%";
    }

    if (summaryDetailEl) {
        let fc = data.face_count || 0;
        if (fc > 0) {
            summaryDetailEl.innerText = fc + " face" + (fc !== 1 ? "s" : "") + " • Score: " + data.score + "%";
            summaryDetailEl.className = "stat-detail attentive";
        } else {
            summaryDetailEl.innerText = "Waiting for faces...";
            summaryDetailEl.className = "stat-detail neutral";
        }
    }

});


// Request new data every 2 seconds
setInterval(function() {
    socket.emit("request_data");
}, 2000);


// ============================================
// ANIMATED NUMBER COUNTER
// ============================================

function animateValue(element, start, end, duration) {
    if (start === end) return;

    const range = end - start;
    let startTime = null;

    function step(timestamp) {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * eased);
        element.innerText = current + "%";

        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    }

    window.requestAnimationFrame(step);
}


// ============================================
// SESSION TIMER
// ============================================

let seconds = 0;

setInterval(function() {
    const statusElement = document.getElementById("sessionStatus");
    const timeElement = document.getElementById("sessionTime");

    if (statusElement && timeElement) {
        if (statusElement.innerText === "Active") {
            seconds++;
            let mins = Math.floor(seconds / 60);
            let secs = seconds % 60;
            timeElement.innerText =
                "Session Time: " + String(mins).padStart(2, '0') + ":" + String(secs).padStart(2, '0');
        }
    }
}, 1000);


// ============================================
// FADE-IN ON SCROLL (Intersection Observer)
// ============================================

document.addEventListener("DOMContentLoaded", function() {
    
    const fadeElements = document.querySelectorAll('.fade-in-up');
    
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver(function(entries) {
            entries.forEach(function(entry) {
                if (entry.isIntersecting) {
                    entry.target.style.animationPlayState = 'running';
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        fadeElements.forEach(function(el) {
            observer.observe(el);
        });
    }

    // Escape key closes sidebar
    document.addEventListener("keydown", function(e) {
        if (e.key === "Escape") {
            const sidebar = document.getElementById("sidebar");
            if (sidebar && sidebar.classList.contains("active")) {
                toggleSidebar();
            }
        }
    });
    
});