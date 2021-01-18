import React, { useState, useContext } from "react";

import { BrowserRouter, Switch, Route } from "react-router-dom";
import { Tabs, Tab } from "react-bootstrap";

import UserContext from "../../../../context/UserContext";
import CreateTest from "../CreateTest";
import FetchLogs from "../FetchLogs";
import DisplayAttentionAnalysis from "../DisplayAttentionAnalysis";
import PageNotFound404 from "../../../misc/pageNotFound404";
import CreateNotification from "../CreateNotification";

export default function AdminHomePage() {
	const { userData } = useContext(UserContext);
	console.log(userData);
	const [key, setKey] = useState("home");

	return (
		<>
			{userData.user && userData.user.admin ? (
				<BrowserRouter>
					<Switch>
						<Route exact path="/">
							<Tabs
								id="controlled-tab-example"
								className="bg-dark"
								activeKey={key}
								onSelect={(k) => setKey(k)}>
								<Tab eventKey="home" title="Create Test">
									<CreateTest />
								</Tab>

								<Tab eventKey="profile" title="Check Attention Tracking Logs">
									<FetchLogs />
								</Tab>

								<Tab eventKey="message" title="Send Notification">
									<CreateNotification />
								</Tab>
							</Tabs>
						</Route>
						<Route exact path="/fetchlogs" component={FetchLogs} />
						<Route path="/fetchlogs/:user/:examID" component={DisplayAttentionAnalysis} />
						<Route>
							<PageNotFound404 />
						</Route>
					</Switch>
				</BrowserRouter>
			) : (
				<>
					<h2>You are not authorised as admin</h2>
				</>
			)}
		</>
	);
}
