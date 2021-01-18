import React, { useEffect, useState, useContext } from "react";

import { Line, Pie, Bar } from "react-chartjs-2";
import { Col, Row, Container, Jumbotron, Button, Table } from "react-bootstrap";
import { Link } from "react-router-dom";
import UserContext from "../../../context/UserContext";
import axios from "axios";
import "hammerjs";
import "chartjs-plugin-zoom";
import "./DisplayAttentionAnalysis.css";

import Loader from "../../misc/Loader";
import {
	generatePieChartData,
	generateLineChartData,
	lineOptions,
	generateBarChartData,
} from "../../../utils/generateChartData";
import { getAttentivenessPercent } from "../../../utils/getAnalysisDetails";

function DisplayAttentionAnalysis({ match }) {
	const serverUrl = "http://127.0.0.1:5000";
	const { userData } = useContext(UserContext);
	const { user, examID } = match.params;
	console.log(match);

	const [loading, setLoading] = useState(true);
	const [examData, setExamData] = useState();
	const [lineData, setLineData] = useState(null);
	const [pieData, setPieData] = useState(null);
	const [barData, setBarData] = useState(null);
	const [obtainedMarks, setObtainedMarks] = useState();
	const [attemptStartTime, setAttemptStartTime] = useState();
	const [attemptEndTime, setAttemptEndTime] = useState();
	const [attentivenessPercent, setAttentivenessPercent] = useState();

	useEffect(() => {
		axios
			.post(
				"http://localhost:10000/users/fetchExam/",
				{ examId: examID },
				{
					headers: { "x-auth-token": userData.token },
				}
			)
			.then((response) => {
				const data = response.data;
				setExamData(data);
				console.log("examdata", data);
			})
			.finally(() => {
				axios
					.get(`${serverUrl}/fetch/${user}/${examID}`)
					.then((response) => {
						const data = response.data;
						// setStudentLogs(data);
						setLineData(generateLineChartData(data.attention_logs, data.created_at));
						setPieData(generatePieChartData(data.attention_logs));
						setBarData(generateBarChartData(data.attention_logs));

						setObtainedMarks(data.marks_obtained);
						setAttemptStartTime(data.created_at);
						setAttemptEndTime(data.finished_at);

						setAttentivenessPercent(getAttentivenessPercent(data.attention_logs));
						console.log(data);
					})
					.finally(() => {
						setLoading(false);
					});
			});
	}, [examID, user, userData.token]);

	if (loading) return <Loader />;

	return (
		<>
			<Jumbotron className="jbtron">
				<Link to="/fetchlogs" className="back-button">
					<Button>Go Back</Button>
				</Link>
				<h1 className="text-primary text-center">Attention Analysis</h1>
			</Jumbotron>
			<Container className="my-5">
				<Table striped bordered hover>
					<tbody className="text-center">
						<tr>
							<th>Name of Student</th>
							<td>{user}</td>
						</tr>
						<tr>
							<th>Exam ID</th>
							<td>{examID}</td>
						</tr>
						<tr>
							<th>Time Limit</th>
							<td>{(examData.timelimit / (1000 * 60)).toFixed(2)} mins</td>
						</tr>
						<tr>
							<th>Total Marks</th>
							<td>{examData.totalMarks}</td>
						</tr>
						<tr>
							<th>Marks Obtained</th>
							<td>{obtainedMarks}</td>
						</tr>
						<tr>
							<th>Attentiveness Percentage</th>
							<td>{attentivenessPercent ? Number(attentivenessPercent).toFixed(3) : "-"}%</td>
						</tr>
						<tr>
							<th>Exam Date</th>
							<td>{new Date(examData.examStartTime).toLocaleDateString()}</td>
						</tr>
						<tr>
							<th>Exam Timing</th>
							<td>
								{new Date(examData.examStartTime).toLocaleTimeString()} to{" "}
								{new Date(examData.examEndTime).toLocaleTimeString()}
							</td>
						</tr>
						<tr>
							<th>Student's attempt Timings</th>
							<td>
								{new Date(attemptStartTime).toLocaleTimeString()} to{" "}
								{new Date(attemptEndTime).toLocaleTimeString()}
							</td>
						</tr>
					</tbody>
				</Table>
				<h4 className="text-primary text-center jumbotron">
					The Charts Below shows a visual depiction of Attention Tracking Logs of the student
				</h4>
				{pieData && (
					<Row className="mb-5">
						<Col>
							<h1 className="text-success text-center">Pie Chart</h1>
							<Pie data={pieData} />
						</Col>
					</Row>
				)}
				<hr />
				{lineData && (
					<Row className="mb-5 mx-3">
						<Col>
							<h1 className="text-success text-center">Line Chart</h1>
							<Line data={lineData} options={lineOptions} />
						</Col>
					</Row>
				)}
				<hr />
				{barData && (
					<Row className="mb-5 mx-3">
						<Col>
							<h1 className="text-success text-center">Bar Chart</h1>
							<Bar data={barData} />
						</Col>
					</Row>
				)}
			</Container>
		</>
	);
}

export default DisplayAttentionAnalysis;
// <h1 className="text-primary">Name : {user}</h1>
// 				<h2 className="text-danger">ExamID : {examID}</h2>
