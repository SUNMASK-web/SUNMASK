// Global vars
var activePanel = "about";
var playState = "init";
var songs = {};
var currentSong = "";
var sceneThree = new THREE.Scene();
var cameraThree = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
var renderer = new THREE.WebGLRenderer();
var channelColors = [
    new THREE.Color('red'),
    new THREE.Color('lightblue'),
    new THREE.Color('yellow'),
    new THREE.Color('purple'),
    new THREE.Color('green'),
    new THREE.Color('orange'),
    new THREE.Color('cyan'),
    new THREE.Color('blue'),
    new THREE.Color('lime'),
    new THREE.Color('fuchsia'),
    new THREE.Color('pink'),
    new THREE.Color('white'),
    new THREE.Color('royalblue'),
    new THREE.Color('tomato'),
    new THREE.Color('beige'),
    new THREE.Color('aqua')
]
var activeNotesThree = [];
var activeNotesP5 = [];

// Set up defaults
cameraThree.position.x = 0;
cameraThree.position.y = 0;
cameraThree.position.z = 1;
cameraThree.lookAt(new THREE.Vector3(0,0,0));
renderer.setSize( window.innerWidth, window.innerHeight );
for(var i = 0; i < 16; i++) {
    activeNotesThree[i] = [];
    activeNotesP5[i] = [];
}

// Panel changing
function changePanel(name) {
    activePanel = name;
    $(".panel").hide();
    $(".panel#" + name).show();
    $("nav a.active").removeClass("active");
    $("nav a.link-" + name).addClass("active");
}

// File Uploading
function handleFileSelect(evt) {
    var files = evt.target.files;
    for (var i = 0, f; f = files[i]; i++) {
        if (!(f.type.match('audio/midi') || f.type.match('audio/mid'))) {
            console.log("Invalid: " + f.type);
            alert("Only MIDI files are allowed");
            continue;
        }
        var reader = new FileReader();
        reader.onload = (function(theFile) {
            return function(e) {
                if(songs[escape(theFile.name)]) return;
                songs[escape(theFile.name)] = e.target.result;
                $("#playlist-list").append("<div class=\"playlist-list-item\" onclick=\"playSong('" + escape(theFile.name) + "')\"><span class=\"fa fa-play-circle\"></span> " + escape(theFile.name) + "</div>");
            };
        })(f);
        reader.readAsDataURL(f);
    }
}

// Time formatting for controls
function timeFormatting(n) {
    var minutes = n / 60 >> 0;
    var seconds = String(n - (minutes * 60) >> 0);
    if (seconds.length == 1) seconds = "0" + seconds;
    return minutes + ":" + seconds;
};

// Some stuff for cleaner play/pause
function changeState(state) {
    if(state == playState) {
        return;
    } else if(state == "start") {
        playState = "play";
        MIDI.Player.start();
        $("#play-btn").hide();
        $("#pause-btn").show();
    } else if(state == "end") {
        playState = "stop";
        $("#play-btn").show();
        $("#pause-btn").hide();
    } else if(state == "play") {
        if(playState == "pause") {
            MIDI.Player.resume();
        } else {
            if(currentSong == "") {
                alert("Please select a song from the playlist");
                return;
            } else {
                playSong(currentSong);
            }
        }
        playState = "play";
        $("#play-btn").hide();
        $("#pause-btn").show();
    } else if(state == "pause") {
        playState = "pause";
        MIDI.Player.pause();
        $("#play-btn").show();
        $("#pause-btn").hide();
    }
}

// Start a new song
function playSong(name) {
    clearScene();
    MIDI.Player.loadFile(songs[name], function() { currentSong = name; changeState("start") });
}

// Reset Scene
function clearScene() {
    while(sceneThree.children.length > 0) {
        sceneThree.remove(sceneThree.children[0]);
    }
    for(var i = 0; i < 16; i++) {
        activeNotesThree[i] = [];
        activeNotesP5[i] = [];
    }
    redraw();
}

// Start JS things when page ready
$(document).ready( function() {
    document.getElementById('files').addEventListener('change', handleFileSelect, false);

    MIDI.loadPlugin({
        soundfontUrl: "https://gleitz.github.io/midi-js-soundfonts/FluidR3_GM/",
		instrument: "acoustic_grand_piano",
        onsuccess: function() {
            for(var i = 0; i < 16; i++) {
                if(i != 9)
                    MIDI.programChange(i, 0);
            }
        }
    });
   MIDI.Player.setAnimation(function(data, element) {
        var percent = data.now / data.end;
        var now = data.now >> 0;
        var end = data.end >> 0;
        if (now === end) {
            changeState("end");
        }
        $("#control-time").text(timeFormatting(now) + " / " + timeFormatting(end));
        $("#control-timeline-completed").css('width', (percent * 500) + "px");
	});

    // Visualization
    $("#visualize").append( renderer.domElement );

    MIDI.Player.addListener(function(data) {
        //var now = data.now;
        //var end = data.end;
        var channel = data.channel;
        var message = data.message;
        var note = data.note;
        var velocity = data.velocity;

        if(message == 144) { // noteOn
            noteOn(channel, note, velocity);
        } else if(message == 128) { // noteOff
            noteOff(channel, note);
        } else if(message == 123) { // all notes off
            clearScene();
        } else {
            console.log("?: " + message + ": " + note + "@" + channel);
        }
    });

    animate();
});

function noteOn(channel, note, velocity) {
    while(activeNotesP5[channel].indexOf(note) != -1)
        noteOff(channel, note);
    var ptsGeometry = new THREE.Geometry();
    var ptV = new THREE.Vector3();
    ptV.x = (note / 127) - 0.5;
    ptV.y = (velocity / 127) - 0.5;
    ptV.z = 0;
    ptsGeometry.vertices.push( ptV )
    var ptsMaterial = new THREE.PointsMaterial( { size: 0.01, color: channelColors[channel] } )
    var ptObj = new THREE.Points( ptsGeometry, ptsMaterial );
    sceneThree.add( ptObj );
    activeNotesThree[channel][note] = ptObj;
    activeNotesP5[channel].push(note);
    redrawKey(channel, note);
}

function noteOff(channel, note) {
    if(activeNotesThree[channel][note] != null) {
        sceneThree.remove(activeNotesThree[channel][note]);
        delete activeNotesThree[channel][note];
    }
    delete activeNotesP5[channel][activeNotesP5[channel].indexOf(note)];
    redrawKey(channel, note);
}

// Three.js stuff
function animate() {
    requestAnimationFrame( animate );
    render();
}
function render() {
    renderer.render( sceneThree, cameraThree );
}

// p5.js stuff
function setup() {
    var canvas = createCanvas(windowWidth, windowHeight);
    canvas.parent('pianoroll');
    noLoop();
}
function draw() {
    //console.log('draw');

    for(var i = 0; i < 16; i++) {
    //for(var i = 0; i < 1; i++) {
        for(var j = 0; j < 128; j++) {
            redrawKey(i, j);
        }
    }
}
function redrawKey(channel, note) {
    //console.log("Key: " + note + "@" + channel);
    var yMin = $("header").height();
    var xI = windowWidth;
    var yI = (windowHeight - yMin) / 16;
    var paddingX = 10;
    var paddingY = 5;
    drawKeyPiano(paddingX, yMin + (yI * channel) + paddingY, xI - paddingX, yMin + (yI * (channel+1)) - paddingY, channel, note);
}
function drawKeyPiano(x1, y1, x2, y2, channel, note) {
    //console.log('draw piano for '+ note + '@' + channel + ': (' + x1 + ',' + y1 + ') to (' + x2 + ',' + y2 + ')');
    var xI = (x2 - x1) / 75;
    var octave = (note - (note % 12)) / 12;
    var wholeOctave = 7; // true
    if (octave == 10)
        wholeOctave = 5; // false
    drawKeyOctave(x1 + (xI * 7 * octave), y1, x1 + (xI * 7 * octave) + (xI * wholeOctave), y2, octave, channel, note);
}
function drawKeyOctave(x1, y1, x2, y2, octave, channel, note) {
    //console.log('draw octave for '+ note + '@' + channel + ': (' + x1 + ',' + y1 + ') to (' + x2 + ',' + y2 + ')');
    var wholeNoteNums = [0, 2, 4, 5, 7, 9, 11];
    var halfNoteNums = [1, 3, 6, 8, 10];
    var yI = (y2 - y1) / 2;
    if(octave == 10) {
        var xI = (x2 - x1) / 5;
        var totalWhole = 5;
        var totalHalf = 3;
    } else {
        var xI = (x2 - x1) / 7;
        var totalWhole = 7;
        var totalHalf = 5;
    }
    var asWholeNote = wholeNoteNums.indexOf(note % 12);
    if(asWholeNote != -1) {
        drawKeyKey(x1 + (xI * asWholeNote), y1 + yI, x1 + (xI * (asWholeNote + 1)), y2, note, false, channel);
    } else {
        var asHalfNote = halfNoteNums.indexOf(note % 12);
        drawKeyKey(x1 + (xI * asHalfNote) + (xI / 2) + (asHalfNote > 1 ? xI : 0), y1, x1 + (xI * (asHalfNote + 1)) + (xI / 2) + (asHalfNote > 1 ? xI : 0), y1 + yI, note, true, channel);
    }
}
function drawKeyKey(x1, y1, x2, y2, note, half, channel) {
    //console.log('draw key for '+ note + '@' + channel + ': (' + x1 + ',' + y1 + ') to (' + x2 + ',' + y2 + ')');
    if(activeNotesP5[channel].indexOf(note) != -1) {
        fill(color('aqua'));
    } else if(half) {
        fill(color('dimgrey'));
    } else {
        fill(color('darkgrey'));
    }
    rect(x1, y1, x2 - x1, y2 - y1);
}
