import React, { useEffect, useState } from "react";

import { Link, useLocation } from "react-router-dom";
import { Card, Button, Container, Jumbotron } from "react-bootstrap";

import Axios from "axios";

function FetchLogs() {
	const serverUrl = "http://127.0.0.1:5000";
	const [logs, setLogs] = useState([]);
	const location = useLocation();

	useEffect(() => {
		Axios.get(`${serverUrl}/fetch`).then((response) => {
			const data = response.data;
			console.log(data);
			let logsList = [];
			data.forEach((userLog) => {
				const userLogDetails = {
					name: userLog.name,
					examId: userLog.exam_id,
					creationTime: new Date(userLog.created_at).toLocaleString(),
					finishTime: new Date(
						userLog.attention_logs[userLog.attention_logs.length - 1].timestamp
					).toLocaleString(),
					examTimeLimit: `${userLog.time_limit / (1000 * 60)} mins`,
					attentionLogs: userLog.attention_logs,
				};
				logsList.push(userLogDetails);
			});
			console.log(logsList);
			setLogs(logsList);
		});
	}, []);
	return (
		<>
			<Jumbotron className="jbtron">
				{location.pathname !== "/" ? (
					<Link to="/" className="back-button">
						<Button>Go Back</Button>
					</Link>
				) : null}
				<h1 className="text-primary text-center">Student Attention Tracking Logs</h1>
			</Jumbotron>
			<Container className="shadow-lg p-3 mb-5 bg-white rounded">
				{logs.map((userLog, index) => (
					<Card key={index} className="my-4">
						<Card.Header className="bg-dark">
							<h3 className="text-primary">{userLog.name}</h3>
							<small className="text-danger">Exam ID : {userLog.examId}</small>
						</Card.Header>
						<Card.Body className="bg-light">
							<div>
								<b className="text-success">Attempt Start Time : </b>
								<span className="text-dark">{userLog.creationTime}</span>
							</div>
							<div>
								<b className="text-danger">Attempt End Time : </b>
								<span className="text-dark">{userLog.finishTime}</span>
							</div>
							<div>
								<b className="text-secondary">Time Limit : </b>
								<span className="text-dark">{userLog.examTimeLimit}</span>
							</div>

							<Link to={`/fetchlogs/${userLog.name}/${userLog.examId}`}>
								<Button variant="primary" block>
									Get Attention Log Analysis
								</Button>
							</Link>
						</Card.Body>
					</Card>
				))}
			</Container>
		</>
	);
}

export default FetchLogs;
