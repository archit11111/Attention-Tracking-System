import React, { useEffect, useContext } from "react";

import { Container, Button, Row, Col } from "react-bootstrap";
import { Link, useHistory, useLocation } from "react-router-dom";
import Bottleneck from "bottleneck";
import Axios from "axios";
import Webcam from "react-webcam";

import AuthOptions from "../../../auth/AuthOptions";
import UserContext from "../../../../context/UserContext";
import Loader from "../../../misc/Loader";
import Questions from "./Questions";
import TimeRemainingCounter from "./TimeRemainingCounter";
import { showErrorAlert } from "../../../../utils/alerts";

import "./WebcamCapture.css";

function WebcamCapture(/*{ userName, examID, timelimit }*/) {
	const { userData } = useContext(UserContext);
	const serverUrl = "http://127.0.0.1:5000";
	const limiter = new Bottleneck({
		maxConcurrent: 1,
		minTime: 200,
	});

	const location = useLocation();
	const history = useHistory();
	const examId = location.pathname.substring(6);
	const userName = userData.user.displayName;

	const [examData, setExamData] = React.useState();
	const [timelimit, setTimelimit] = React.useState();
	const [loading, setLoading] = React.useState(true);
	const [loading2, setLoading2] = React.useState(true);
	const [startTime, setStartTime] = React.useState(-1);
	const [timeExpired, setTimeExpired] = React.useState(false);
	const [isCalibrated, setIsCalibrated] = React.useState(false);
	const [captureStarted, setCaptureStarted] = React.useState(false);

	useEffect(() => {
		console.log("sdadsasdasd");
		Axios.post(
			"http://127.0.0.1:10000/users/fetchExam",
			{ examId },
			{
				headers: { "x-auth-token": userData.token },
			}
		)
			.then((res) => {
				const data = res.data;
				if (new Date(data.examStartTime).getTime() > Date.now()) {
					showErrorAlert("Oops...", "Exam not started yet! Please try again later");
					history.push("/");
				}
				setExamData(data);
				console.log("examData", data);
				setTimelimit(data.timelimit);
				console.log("timelimit", data.timelimit);
			})
			.catch(() => {
				showErrorAlert();
				history.push("/");
			});
	}, [examId, history, userData.token]);

	// const [remainingTime, setRemainingTime] = React.useState(null);

	// get all values on reload -- case if test already started
	useEffect(() => {
		Axios.post(`${serverUrl}/time`, { user: userName, examID: examId })
			.then((res) => {
				const createdAt = res.data.created_at;
				const finishedAt = res.data.finished_at;

				if (finishedAt !== "-") {
					// finished_at is only logged once user ends test or time expires
					console.log(finishedAt);
					setTimeExpired(true);
					return;
				}

				// const calibratedValues = res.data.calibrated_values;
				console.log(createdAt);

				const timestamp = new Date(createdAt).getTime();
				setStartTime(timestamp);
				// console.log('startTime', timestamp)
				// setCalibrationObj(calibratedValues);
			})
			.catch((err) => console.log(err))
			.finally(() => setLoading(false));
	}, [examId, userName]);

	const webcamRef = React.useRef(null);

	const postImageFrames = () => {
		return new Promise((resolve, reject) => {
			if (!webcamRef || !webcamRef.current) return;
			const imageSrc = webcamRef.current.getScreenshot();

			Axios.post(serverUrl, {
				imageSrc,
				user: userName,
				examID: examId,
				timeLimit: timelimit,
			})
				.then((res) => {
					resolve(res);
				})
				.catch((err) => {
					console.log(err);
					reject();
				});
		});
	};

	const captureHandler = () => {
		if (/*remainingTime === 0 ||*/ timeExpired || document.getElementById("webcam-frame") === null) return;
		limiter
			.schedule(() => postImageFrames())

			.then((res) => {
				if (loading2) {
					setLoading2(false);
				}
				const { success, data } = res.data;

				setIsCalibrated(!data["isCalibrating"] === true);
				console.log(data);
				let text = "";
				if (success === false) {
					text = "Failed";
				} else {
					if (data["isCalibrating"] === true) text = "calibrating";
					else text = data["isAttentive"] === true ? "Attentive" : "Inattentive";
				}
				document.getElementById("res").innerHTML = text;
			})
			.catch((err) => {
				console.log("err", err);
			})
			.finally(() => captureHandler());
	};

	const startCaptureHandler = () => {
		alert("Look at your camera for first 5 seconds to allow for calibration");
		// setLoading2(true);
		setCaptureStarted(true);
		if (startTime === -1) setStartTime(Date.now());

		captureHandler();
	};

	const reCalibrateHandler = () => {
		Axios.post(`${serverUrl}/recalibrate`, { user: userName, examID: examId }).then((res) => {
			console.log("re-calibration started");
		});
	};

	const endTestHandler = () => {
		setTimeExpired(true);
		Axios.post(`${serverUrl}/end`, { user: userName, examID: examId })
			.then((response) => {
				const data = response.data;
				console.log(data === "success" ? "test ended" : "err");
			})
			.catch((err) => console.log(err));
	};

	if (loading) return <Loader />;

	if (timeExpired) {
		return (
			<Container className="m-auto text-center">
				<h1>Your test has ended!</h1>
				<div className="test-end-logout btn btn-danger btn-lg mr-3" onClick={() => history.push("/")}>
					<AuthOptions />
				</div>
				<Link to="/">
					<Button variant="success" size="lg">
						Go Home
					</Button>
				</Link>
			</Container>
		);
	}

	return (
		<Container fluid>
			<Row>
				{startTime === -1 || !captureStarted ? (
					<Col className="question-col">
						<h1 className="text-success text-center">Important Instructions:</h1>
						<ol className="instructions">
							<li>
								The first few seconds of the web-cam feed will be use to calibrate baseline values for
								your normal attentive state. Please ensure that you maintain your normal attentive
								posture for first 5 seconds as it will be used to track you further.
							</li>
							<li>
								You have the option to recalibrate the tracker if you feel that it was not calibrated
								properly.
							</li>
						</ol>
					</Col>
				) : loading2 === true ? (
					<Loader />
				) : (
					examData && (
						<Col className="question-col">
							<Questions examData={examData} />
						</Col>
					)
				)}
				<Col className="webcam-col">
					<div className="webcam">
						<Webcam
							audio={false}
							ref={webcamRef}
							forceScreenshotSourceSize
							videoConstraints={{
								height: 480,
								width: 640,
							}}
							className="webcam-frame"
							id="webcam-frame"
						/>
						{isCalibrated ? (
							<Button onClick={reCalibrateHandler} className="recalibrate-button">
								Re-Calibrate
							</Button>
						) : null}
						<div className="details-container">
							<div id="result">
								<span className="text-success">Result : </span>
								<span id="res">None</span>
							</div>
							<div id="time-remaining">
								<span className="text-danger">Time Left : </span>
								<TimeRemainingCounter
									startTime={startTime}
									timeLimit={timelimit}
									// setTimeExpired={setTimeExpired}
									endTestHandler={endTestHandler}
								/>
							</div>
							{startTime !== -1 ? (
								<Button id="end-test" variant="danger" onClick={endTestHandler}>
									<span>END TEST</span>
								</Button>
							) : null}
						</div>
					</div>
				</Col>
			</Row>
			{startTime === -1 || !captureStarted ? (
				<Button onClick={startCaptureHandler} className="start-button">
					Start
				</Button>
			) : null}
		</Container>
	);
}

export default WebcamCapture;
